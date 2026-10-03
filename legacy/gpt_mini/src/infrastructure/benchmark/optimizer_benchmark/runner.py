"""GPU-only training and fair, persistent hyperparameter comparisons."""

from gpt_mini.infrastructure.benchmark.paths import benchmark_path as _bench_path
from gpt_mini.infrastructure.benchmark.provenance import source_hashes as _source_hashes, git_head as _git_head
from gpt_mini.infrastructure.benchmark.suites import test_suite as _test_suite

from dataclasses import asdict
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import subprocess
import time
import traceback
import unittest

import torch
from torch.nn import functional as F

from gpt_mini.gpt_mini import Config
from .attention import make_model
from .data import TextData, evaluation_batches, select_learning_rates, training_starts, windows
from .registry import METHODS, make_optimizer
from .report import write_report
from .storage import read_results, write_json, write_summary


def run_tests(output):
    suite = _test_suite(('optimizer_benchmark',))
    with output.open("w") as stream:
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    if not result.wasSuccessful() or result.skipped:
        raise RuntimeError(f"All CPU and CUDA tests must pass before training; see {output}")
    return {"tests": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
            "skipped": len(result.skipped), "log_sha256": hashlib.sha256(output.read_bytes()).hexdigest()}


@torch.no_grad()
def evaluate(model, tokens, context, batch):
    model.eval()
    total = torch.zeros((), device=tokens.device)
    count = 0
    for inputs, targets in evaluation_batches(tokens, context, batch):
        logits = model(inputs)
        total += F.cross_entropy(logits.flatten(0, 1), targets.flatten(), reduction="sum")
        count += targets.numel()
    model.train()
    if not count:
        raise ValueError("Held-out split is too short for evaluation")
    loss = total.item() / count
    if not math.isfinite(loss):
        raise FloatingPointError("Nonfinite held-out loss")
    return loss


def state_hash(model):
    digest = hashlib.sha256()
    for name, p in model.named_parameters():
        digest.update(name.encode())
        digest.update(p.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def run_id(attention, method, rate, seed, phase, steps):
    return f"{attention}:{method}:{rate.hex()}:{seed}:{phase}:{steps}"


def train_one(config, data, attention, method, rate, seed, phase, steps, batch,
              *, checkpoints=None, curve_every=250):
    started = time.perf_counter()
    model = make_model(config, attention, seed, "cuda")
    initial_hash = state_hash(model)
    optimizer = make_optimizer(method, model, rate)
    starts_cpu = training_starts(len(data.train), config.max_seq_len, batch, steps, seed)
    batch_hash = hashlib.sha256(starts_cpu.numpy().tobytes()).hexdigest()
    starts = starts_cpu.to("cuda")
    result = {"id": run_id(attention, method, rate, seed, phase, steps),
              "attention": attention, "method": method, "lr": rate, "seed": seed,
              "phase": phase, "steps": steps, "tokens_seen": steps * batch * config.max_seq_len,
              "initial_sha256": initial_hash, "batch_sha256": batch_hash, "status": "ok",
              "validation_loss": None, "test_loss": None, "test_ppl": None, "curves": []}
    torch.cuda.reset_peak_memory_stats()
    train_seconds = 0.0
    step_events = []
    try:
        if phase != "calibration":
            baseline = evaluate(model, data.validation, config.max_seq_len, batch)
            result["curves"].append({"step": 0, "validation_loss": baseline})
        for step in range(steps):
            torch.cuda.synchronize()
            train_start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            inputs, targets = windows(data.train, starts[step], config.max_seq_len)
            logits = model(inputs)
            loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
            if not math.isfinite(loss.item()):
                raise FloatingPointError(f"Nonfinite training loss before update {step + 1}")
            loss.backward()
            before, after = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            before.record()
            optimizer.step()
            after.record()
            torch.cuda.synchronize()
            train_seconds += time.perf_counter() - train_start
            step_events.append((before, after))
            if phase == "final" and ((step + 1) % curve_every == 0 or step + 1 == steps):
                validation = evaluate(model, data.validation, config.max_seq_len, batch)
                result["curves"].append({"step": step + 1, "train_loss": loss.item(),
                                         "validation_loss": validation, "train_seconds": train_seconds})
        if phase != "calibration":
            result["validation_loss"] = evaluate(model, data.validation, config.max_seq_len, batch)
            if not math.isfinite(result["validation_loss"]):
                raise FloatingPointError("Nonfinite final validation loss")
        if phase == "final":
            result["test_loss"] = evaluate(model, data.test, config.max_seq_len, batch)
            if not math.isfinite(result["test_loss"]):
                raise FloatingPointError("Nonfinite final test loss")
            result["test_ppl"] = math.exp(result["test_loss"])
            if seed == 0 and checkpoints is not None:
                path = checkpoints / f"{attention}_{method}_seed0.pt"
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"config": asdict(config), "attention": attention, "method": method,
                            "lr": rate, "seed": seed, "steps": steps,
                            "model": {k: v.detach().cpu() for k, v in model.state_dict().items()}}, path)
                result["checkpoint"] = str(path)
    except (FloatingPointError, RuntimeError) as error:
        result["status"] = "failed"
        result["error"] = str(error)
        result["traceback"] = traceback.format_exc()
        for field in ["validation_loss", "test_loss", "test_ppl"]:
            if result[field] is not None and not math.isfinite(result[field]):
                result[field] = None
    finally:
        result["completed_steps"] = optimizer.steps
        result["train_seconds"] = train_seconds
        result["optimizer_ms"] = sum(a.elapsed_time(b) for a, b in step_events) / max(len(step_events), 1)
        result["peak_memory_mib"] = torch.cuda.max_memory_allocated() / 2 ** 20
        result["peak_reserved_mib"] = torch.cuda.max_memory_reserved() / 2 ** 20
        result["guard_acceptance"] = optimizer.guard_accepted / max(optimizer.steps, 1) if optimizer.guarded else None
        result["total_seconds"] = time.perf_counter() - started
        optimizer.close()
        del optimizer, model, starts
        gc.collect()
        torch.cuda.empty_cache()
    return result


def source_hashes():
    return _source_hashes()


def log_progress(directory, message):
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    line = f"{stamp} {message}"
    print(line, flush=True)
    with (directory / "progress.log").open("a") as stream:
        stream.write(line + "\n")


def launch(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; this experiment never falls back to CPU")
    directory = Path(args.output)
    directory.mkdir(parents=True, exist_ok=True)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tests = run_tests(directory / "tests.log")
    torch.set_num_threads(4)
    log_progress(directory, f"All {tests['tests']} CPU/CUDA tests passed")
    data = TextData.load(args.data, "cuda")
    config = Config(vocab_size=len(data.characters), n_layers=args.layers, n_heads=4,
                    d_model=args.width, d_ff=4 * args.width, max_seq_len=args.context)
    methods = [m for m in METHODS if not args.only or m.name in args.only]
    if not methods or args.only and set(args.only) != {m.name for m in methods}:
        raise ValueError("Unknown method selection")
    props = torch.cuda.get_device_properties(0)
    protocol = {"config": asdict(config), "batch": args.batch, "screen_steps": args.screen_steps,
                "final_steps": args.steps, "seeds": args.seeds, "attention": args.attention,
                "methods": [asdict(m) for m in methods], "data_sha256": data.sha256,
                "data_boundaries": data.boundaries, "characters": data.characters,
                "source_hashes": source_hashes(), "initialization": "Normal(0,0.02) unique matrices",
                "dtype": "float32", "tf32": False, "deterministic_algorithms": True}
    encoded = json.dumps(protocol, sort_keys=True).encode()
    protocol_hash = hashlib.sha256(encoded).hexdigest()
    metadata = {"protocol": protocol, "protocol_sha256": protocol_hash, "tests": tests,
                "environment": {"gpu": props.name, "gpu_total_bytes": props.total_memory,
                                "torch": torch.__version__, "cuda": torch.version.cuda,
                                "python": platform.python_version(), "platform": platform.platform(),
                                "git_head": _git_head()},
                "created_utc": datetime.now(timezone.utc).isoformat()}
    if args.calibrate:
        write_json(directory / "calibration_metadata.json", metadata)
        measurements = []
        for attention in args.attention:
            for method in methods:
                row = train_one(config, data, attention, method.name, method.rates[1], 0,
                                "calibration", 12, args.batch)
                measurements.append(row)
                write_json(directory / "calibration.json", measurements)
                log_progress(directory, f"calibration {attention} {method.name}: {row['status']}, "
                             f"{1000 * row['train_seconds'] / 12:.1f} ms/step, "
                             f"{row['peak_memory_mib']:.0f} MiB")
        return
    metadata_path = directory / "metadata.json"
    if metadata_path.exists():
        previous = json.loads(metadata_path.read_text())
        if previous["protocol_sha256"] != protocol_hash:
            raise RuntimeError("Existing results use a different protocol/source; choose a new output directory")
    else:
        write_json(metadata_path, metadata)
    raw = directory / "runs.jsonl"
    results = read_results(raw)
    completed = {r["id"] for r in results}
    checkpoints = Path(_bench_path('experiments/runs/optimizer_benchmark')) / directory.name

    def execute(jobs, phase):
        random.Random(1729 if phase == "screen" else 2718).shuffle(jobs)
        for index, (attention, method, rate, seed, steps) in enumerate(jobs, 1):
            identifier = run_id(attention, method, rate, seed, phase, steps)
            if identifier in completed:
                continue
            log_progress(directory, f"{phase} {index}/{len(jobs)} START {attention} {method} lr={rate:g} seed={seed}")
            row = train_one(config, data, attention, method, rate, seed, phase, steps, args.batch,
                            checkpoints=checkpoints)
            row["protocol_sha256"] = protocol_hash
            with raw.open("a") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
            results.append(row)
            completed.add(identifier)
            write_summary(directory, results, args.seeds)
            loss = row["validation_loss"]
            log_progress(directory, f"{phase} {index}/{len(jobs)} END {attention} {method}: "
                         f"{row['status']}, val={loss}, train={row['train_seconds']:.1f}s")

    screen = [(a, m.name, lr, 0, args.screen_steps) for a in args.attention
              for m in methods for lr in m.rates]
    execute(screen, "screen")
    selected = {a: select_learning_rates([r for r in results if r["attention"] == a],
                                         [m.name for m in methods]) for a in args.attention}
    write_json(directory / "selected_rates.json", selected)
    final = [(a, m.name, selected[a][m.name], s, args.steps) for a in args.attention
             for m in methods if selected[a][m.name] is not None for s in args.seeds]
    execute(final, "final")
    write_summary(directory, results, args.seeds)
    write_report(directory)
    log_progress(directory, "COMPLETE: both attention comparisons and all raw results saved")
