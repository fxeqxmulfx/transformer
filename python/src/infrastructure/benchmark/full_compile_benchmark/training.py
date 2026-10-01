"""Train to a validation stopping decision and restore the exact best model."""

from dataclasses import asdict
import gc
import hashlib
import math
from pathlib import Path
import time
import traceback

import torch

from gpt_mini.infrastructure.benchmark.optimizer_benchmark.attention import make_model
from gpt_mini.infrastructure.benchmark.optimizer_benchmark.data import training_starts
from gpt_mini.infrastructure.benchmark.optimizer_benchmark.runner import state_hash
from gpt_mini.infrastructure.benchmark.optimizer_benchmark.storage import write_json
from .step import FullStep
from gpt_mini.infrastructure.benchmark.compiled_benchmark.telemetry import snapshot
from gpt_mini.infrastructure.benchmark.patience_benchmark.stopping import EarlyStopper


def finite_tree(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: finite_tree(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [finite_tree(item) for item in value]
    return value


def train_one(config, data, attention, method, rate, seed, batch, stopping, artifacts,
              *, device="cuda", progress=None):
    torch.compiler.reset()
    from torch._dynamo.utils import counters
    counters.clear()
    started = time.perf_counter()
    cuda = torch.device(device).type == "cuda"
    artifacts = Path(artifacts)
    artifacts.mkdir(parents=True, exist_ok=True)
    checkpoint = artifacts / f"{attention}_{method}_seed{seed}_best.pt"
    curve_path = artifacts / f"{attention}_{method}_seed{seed}_curve.json"
    model = make_model(config, attention, seed, device)
    runtime = FullStep(model, attention, method, rate, seed, batch, stopping.max_steps)
    optimizer = runtime.opt
    compiler_evaluations = []
    starts_cpu = training_starts(len(data.train), config.max_seq_len, batch, stopping.max_steps, seed)
    starts = starts_cpu.to(device)
    stopper = EarlyStopper(stopping)
    row = {"id": f"{attention}:{method}:{seed}", "attention": attention, "method": method,
           "lr": rate, "seed": seed, "status": "ok", "stop_reason": None,
           "actual_steps": 0, "best_step": None, "validation_loss": None,
           "last_validation_loss": None, "test_loss": None, "test_ppl": None,
           "test_evaluations": 0, "curves": [], "initial_sha256": state_hash(model),
           "batch_plan_sha256": hashlib.sha256(starts_cpu.numpy().tobytes()).hexdigest()}
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    train_seconds = 0.0
    events = []

    def check(step, train_loss=None):
        try:
            validation = runtime.evaluate(data.validation, batch)
        except FloatingPointError:
            validation = math.nan
        compiler_evaluations.append({"step": step, **snapshot()})
        decision = stopper.observe(step, validation)
        row["last_validation_loss"] = validation if math.isfinite(validation) else None
        row["curves"].append({"step": step, "validation_loss": row["last_validation_loss"],
                              "train_loss": train_loss, "new_best": decision.new_best,
                              "bad_checks": stopper.bad_checks,
                              "divergent_checks": stopper.divergent_checks,
                              "train_seconds": train_seconds, "stop_reason": decision.reason})
        if decision.new_best:
            payload = {"config": asdict(config), "attention": attention, "method": method,
                       "lr": rate, "seed": seed, "best_step": step, "validation_loss": validation,
                       "model": {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}}
            temporary = checkpoint.with_suffix(".pt.tmp")
            torch.save(payload, temporary)
            temporary.replace(checkpoint)
        write_json(curve_path, row["curves"])
        if progress is not None:
            progress(f"check {attention} {method} seed={seed} step={step} "
                     f"val={row['last_validation_loss']} best={stopper.best_loss} "
                     f"bad={stopper.bad_checks}/{stopping.patience} "
                     f"FX={compiler_evaluations[-1]['unique_fx_graphs']} "
                     f"CUDA_nodes={compiler_evaluations[-1]['recorded_graph_nodes']}")
        if decision.should_stop:
            row["stop_reason"] = decision.reason
        return decision.should_stop

    try:
        stopped = check(0)
        for step in range(1, stopping.max_steps + 1):
            if stopped:
                break
            if cuda:
                torch.cuda.synchronize()
            train_start = time.perf_counter()
            before, after = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            before.record()
            loss = runtime.train(data.train, starts)
            after.record()
            torch.cuda.synchronize()
            loss_value = loss.item()
            if not math.isfinite(loss_value):
                row["stop_reason"] = "nonfinite_training"
                row["terminal_error"] = {"attempted_step": step, "loss": None}
                break
            events.append((before, after))
            train_seconds += time.perf_counter() - train_start
            row["actual_steps"] = step
            if step % stopping.every == 0 or step == stopping.max_steps:
                stopped = check(step, loss_value)
    except (FloatingPointError, RuntimeError) as error:
        row.update(stop_reason="runtime_error", error=str(error), traceback=traceback.format_exc())
    try:
        if row["stop_reason"] in {"nonfinite_training", "nonfinite_validation", "runtime_error"}:
            row["status"] = "recovered"
        if stopper.best_step is None:
            raise RuntimeError("No finite validation checkpoint is available")
        payload = torch.load(checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(payload["model"])
        row["best_step"] = stopper.best_step
        row["validation_loss"] = runtime.evaluate(data.validation, batch)
        if not math.isclose(row["validation_loss"], stopper.best_loss, rel_tol=0, abs_tol=1e-9):
            raise RuntimeError("Restored checkpoint does not reproduce its validation loss")
        row["test_evaluations"] += 1
        row["test_loss"] = runtime.evaluate(data.test, batch)
        row["test_ppl"] = math.exp(row["test_loss"])
        row["best_model_sha256"] = state_hash(model)
        row["checkpoint"] = str(checkpoint)
        row["checkpoint_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        del payload
    except (FloatingPointError, RuntimeError) as error:
        row.update(status="failed", recovery_error=str(error))
    finally:
        row.update(train_seconds=train_seconds, total_seconds=time.perf_counter() - started,
                   optimizer_ms=None,
                   whole_step_ms=sum(a.elapsed_time(b) for a, b in events) / max(len(events), 1),
                   peak_memory_mib=torch.cuda.max_memory_allocated() / 2**20 if cuda else None,
                   tokens_seen=row["actual_steps"] * batch * config.max_seq_len,
                   batch_sha256=hashlib.sha256(starts_cpu[:row["actual_steps"]].numpy().tobytes()).hexdigest(),
                   guard_acceptance=runtime.accepted.item() / max(runtime.counter.item(), 1) if optimizer.guarded else None,
                   magma=magma_diagnostics(optimizer),
                   compile={**runtime.compilation_settings, "evaluations": compiler_evaluations, "final": snapshot()},
                   peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20)
        runtime.close()
        del runtime, optimizer, model, starts
        torch.compiler.reset()
        gc.collect()
        if cuda:
            torch.cuda.empty_cache()
    return finite_tree(row)


def magma_diagnostics(optimizer):
    if not hasattr(optimizer, "masked_parameters"):
        return None
    blocks = {}
    for p in optimizer.param_groups[0]["params"]:
        if p not in optimizer.masked_parameters:
            continue
        state = optimizer.state[p]
        draws, kept = state["draws"].item(), state["kept"].item()
        blocks[optimizer.names[p]] = {"draws": draws, "kept": kept,
            "survival_fraction": kept / draws if draws else None,
            "mean_damping": state["scale_sum"].item() / draws if draws else None,
            "mean_cosine": state["cosine_sum"].item() / draws if draws else None,
            "final_damping": state["scale"].item()}
    draws = sum(b["draws"] for b in blocks.values())
    return {"blocks": blocks, "mask_draws": draws,
        "survival_fraction": sum(b["kept"] for b in blocks.values()) / draws if draws else None}
