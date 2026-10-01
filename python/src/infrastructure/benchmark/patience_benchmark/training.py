"""Train to a validation stopping decision and restore the exact best model."""

from dataclasses import asdict
import gc
import hashlib
import math
from pathlib import Path
import time
import traceback

import torch
from torch.nn import functional as F

from gpt_mini.infrastructure.benchmark.optimizer_benchmark.attention import make_model
from gpt_mini.infrastructure.benchmark.optimizer_benchmark.data import training_starts, windows
from gpt_mini.infrastructure.benchmark.optimizer_benchmark.runner import evaluate, state_hash
from gpt_mini.infrastructure.benchmark.optimizer_benchmark.storage import write_json
from .registry import make_optimizer
from .stopping import EarlyStopper


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
    started = time.perf_counter()
    cuda = torch.device(device).type == "cuda"
    artifacts = Path(artifacts)
    artifacts.mkdir(parents=True, exist_ok=True)
    checkpoint = artifacts / f"{attention}_{method}_seed{seed}_best.pt"
    curve_path = artifacts / f"{attention}_{method}_seed{seed}_curve.json"
    model = make_model(config, attention, seed, device)
    optimizer = make_optimizer(method, model, rate, seed)
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
    train_seconds, optimizer_cpu_seconds = 0.0, 0.0
    events = []

    def check(step, train_loss=None):
        try:
            validation = evaluate(model, data.validation, config.max_seq_len, batch)
        except FloatingPointError:
            validation = math.nan
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
                     f"bad={stopper.bad_checks}/{stopping.patience}")
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
            optimizer.zero_grad(set_to_none=True)
            inputs, targets = windows(data.train, starts[step - 1], config.max_seq_len)
            loss = F.cross_entropy(model(inputs).flatten(0, 1), targets.flatten())
            loss_value = loss.item()
            if not math.isfinite(loss_value):
                row["stop_reason"] = "nonfinite_training"
                row["terminal_error"] = {"attempted_step": step, "loss": None}
                break
            loss.backward()
            if cuda:
                before, after = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                before.record()
            optimizer_start = time.perf_counter()
            optimizer.step()
            optimizer_cpu_seconds += time.perf_counter() - optimizer_start
            if cuda:
                after.record()
                torch.cuda.synchronize()
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
        row["validation_loss"] = evaluate(model, data.validation, config.max_seq_len, batch)
        if not math.isclose(row["validation_loss"], stopper.best_loss, rel_tol=0, abs_tol=1e-9):
            raise RuntimeError("Restored checkpoint does not reproduce its validation loss")
        row["test_evaluations"] += 1
        row["test_loss"] = evaluate(model, data.test, config.max_seq_len, batch)
        row["test_ppl"] = math.exp(row["test_loss"])
        row["best_model_sha256"] = state_hash(model)
        row["checkpoint"] = str(checkpoint)
        row["checkpoint_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        del payload
    except (FloatingPointError, RuntimeError) as error:
        row.update(status="failed", recovery_error=str(error))
    finally:
        row.update(train_seconds=train_seconds, total_seconds=time.perf_counter() - started,
                   optimizer_ms=sum(a.elapsed_time(b) for a, b in events) / max(len(events), 1)
                   if cuda else 1000 * optimizer_cpu_seconds / max(row["actual_steps"], 1),
                   peak_memory_mib=torch.cuda.max_memory_allocated() / 2**20 if cuda else None,
                   tokens_seen=row["actual_steps"] * batch * config.max_seq_len,
                   batch_sha256=hashlib.sha256(starts_cpu[:row["actual_steps"]].numpy().tobytes()).hexdigest(),
                   guard_acceptance=optimizer.guard_accepted / max(optimizer.steps, 1) if optimizer.guarded else None,
                   magma=optimizer.diagnostics() if hasattr(optimizer, "diagnostics") else None)
        optimizer.close()
        del optimizer, model, starts
        gc.collect()
        if cuda:
            torch.cuda.empty_cache()
    return finite_tree(row)
