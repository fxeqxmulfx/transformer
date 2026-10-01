"""Test, calibrate, screen, train, audit, and save the predeclared follow-up."""

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import random
import subprocess
import unittest

import torch

from experiments.gpt_mini import Config
from experiments.optimizer_benchmark.data import TextData, select_learning_rates
from experiments.optimizer_benchmark.runner import log_progress, run_id
from experiments.optimizer_benchmark.storage import read_results, write_json, write_summary
from .protocol import baseline, check_compatible, check_pairing, source_hashes
from .registry import METHODS
from .report import write_report
from .training import train_one
from .validation import validate


def run_tests(directory):
    loader = unittest.TestLoader()
    suite = unittest.TestSuite([
        loader.discover("experiments/optimizer_benchmark/tests", top_level_dir="."),
        loader.discover("experiments/magma_benchmark/tests", top_level_dir="."),
        loader.loadTestsFromName("experiments.test_optimizer_parameter_groups")])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    path = directory / f"tests_{stamp}.log"
    with path.open("w") as stream:
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    if not result.wasSuccessful() or result.skipped:
        raise RuntimeError(f"CPU/CUDA tests must all pass before training: {path}")
    return {"tests": result.testsRun, "failures": 0, "errors": 0, "skipped": 0,
            "log": str(path), "log_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def launch(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this experiment")
    directory = Path(args.output)
    directory.mkdir(parents=True, exist_ok=True)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.validate_only:
        text = validate(directory)
        write_report(directory)
        print(json.dumps(text, indent=2), flush=True)
        return
    tests = run_tests(directory)
    torch.set_num_threads(4)
    log_progress(directory, f"All {tests['tests']} CPU/CUDA tests passed before training")
    old_rows, old_metadata, provenance = baseline()
    common = old_metadata["protocol"]
    data = TextData.load("experiments/tinyshakespeare.txt", "cuda")
    config = Config(**common["config"])
    protocol = {field: common[field] for field in (
        "batch", "screen_steps", "final_steps", "seeds", "attention", "initialization",
        "dtype", "tf32", "deterministic_algorithms")}
    protocol.update(config=asdict(config), data_sha256=data.sha256,
                    data_boundaries=list(data.boundaries), characters=data.characters,
                    methods=[asdict(m) for m in METHODS], source_hashes=source_hashes(),
                    baseline=provenance, recipe={"algorithm": "Algorithm 1; s*m, no 1/p",
                    "p": 0.5, "tau": 2.0, "scale_beta": 0.9, "initial_scale": 0.5,
                    "zero_cosine": 0.0, "mask_seed": "20000+model_seed",
                    "blocks": "whole attention/FFN matrices; fused QKV is one block"})
    check_compatible(protocol, common)
    fingerprint = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    props = torch.cuda.get_device_properties(0)
    metadata = {"protocol": protocol, "protocol_sha256": fingerprint, "tests": tests,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "environment": {"gpu": props.name, "gpu_total_bytes": props.total_memory,
                    "torch": torch.__version__, "cuda": torch.version.cuda,
                    "python": platform.python_version(), "platform": platform.platform(),
                    "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                    "nvidia_smi": subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                                                          "--format=csv,noheader"], text=True).strip()}}
    checkpoint_dir = Path("experiments/runs/magma_benchmark") / directory.name
    if args.calibrate:
        write_json(directory / "calibration_metadata.json", metadata)
        rows = []
        for attention in common["attention"]:
            for method in METHODS:
                row = train_one(config, data, attention, method.name, method.rates[1], 0,
                                "calibration", 12, common["batch"], checkpoints=checkpoint_dir)
                rows.append(row)
                write_json(directory / "calibration.json", rows)
                log_progress(directory, f"calibration {attention} {method.name}: {row['status']}, "
                             f"{row['peak_memory_mib']:.0f} MiB, {1000 * row['train_seconds'] / 12:.1f} ms/step")
        if any(r["status"] != "ok" for r in rows):
            raise RuntimeError("Calibration failed; inspect saved rows")
        return
    path = directory / "metadata.json"
    if path.exists():
        if json.loads(path.read_text())["protocol_sha256"] != fingerprint:
            raise RuntimeError("Changed data/source/protocol; use a new output directory")
    else:
        write_json(path, metadata)
    raw = directory / "runs.jsonl"
    rows = read_results(raw)
    completed = {r["id"] for r in rows}

    def execute(jobs, phase):
        random.Random(1729 if phase == "screen" else 2718).shuffle(jobs)
        for index, (attention, method, rate, seed, steps) in enumerate(jobs, 1):
            identifier = run_id(attention, method, rate, seed, phase, steps)
            if identifier in completed:
                continue
            log_progress(directory, f"{phase} {index}/{len(jobs)} START {attention} {method} lr={rate:g} seed={seed}")
            row = train_one(config, data, attention, method, rate, seed, phase, steps,
                            common["batch"], checkpoints=checkpoint_dir)
            check_pairing([row], old_rows)
            row["protocol_sha256"] = fingerprint
            with raw.open("a") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
            rows.append(row)
            completed.add(identifier)
            write_summary(directory, rows, common["seeds"])
            log_progress(directory, f"{phase} {index}/{len(jobs)} END {attention} {method}: "
                         f"{row['status']}, val={row['validation_loss']}, train={row['train_seconds']:.1f}s")

    execute([(a, m.name, lr, 0, common["screen_steps"]) for a in common["attention"]
             for m in METHODS for lr in m.rates], "screen")
    selected = {a: select_learning_rates([r for r in rows if r["attention"] == a],
                                         [m.name for m in METHODS]) for a in common["attention"]}
    write_json(directory / "selected_rates.json", selected)
    execute([(a, m.name, selected[a][m.name], s, common["final_steps"]) for a in common["attention"]
             for m in METHODS if selected[a][m.name] is not None for s in common["seeds"]], "final")
    write_summary(directory, rows, common["seeds"])
    validate(directory)
    write_report(directory)
    log_progress(directory, "COMPLETE: raw results, checkpoints, audits, and all-method rankings saved")
