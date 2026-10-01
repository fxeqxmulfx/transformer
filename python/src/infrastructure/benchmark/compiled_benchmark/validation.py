"""Replay each stopping controller and verify every saved best model on CUDA."""

from gpt_mini.infrastructure.benchmark.paths import artifact_path as _artifact_path, DATA_FILE as _data_file
from gpt_mini.infrastructure.benchmark.provenance import source_snapshot_matches as _snapshot_matches, artifact_snapshot_matches as _artifact_snapshot_matches

import gc
import hashlib
import json
import math
from pathlib import Path

import torch

from gpt_mini.gpt_mini import Config
from gpt_mini.infrastructure.benchmark.optimizer_benchmark.attention import make_model
from gpt_mini.infrastructure.benchmark.optimizer_benchmark.data import TextData, training_starts
from gpt_mini.infrastructure.benchmark.optimizer_benchmark.runner import evaluate, state_hash
from gpt_mini.infrastructure.benchmark.optimizer_benchmark.storage import read_results, write_json
from gpt_mini.infrastructure.benchmark.patience_benchmark.protocol import check_initial, previous_runs, prior_fingerprints
from .protocol import source_hashes, preflight_hashes
from .backend import compile_model, graph_counts
from gpt_mini.infrastructure.benchmark.patience_benchmark.registry import make_optimizer, selected_rates
from gpt_mini.infrastructure.benchmark.patience_benchmark.stopping import EarlyStopper, StopConfig, choose_best_step
from gpt_mini.infrastructure.benchmark.patience_benchmark.storage import summarize


def validate(directory, *, complete=True):
    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text())
    p = metadata["protocol"]
    fingerprint = hashlib.sha256(json.dumps(p, sort_keys=True).encode()).hexdigest()
    if fingerprint != metadata["protocol_sha256"] or not _snapshot_matches(p["source_hashes"]):
        raise ValueError("Protocol/source fingerprint mismatch")
    if not _artifact_snapshot_matches(p["prior_artifacts"]):
        raise ValueError("Earlier measurements changed")
    if not p["compile"] or not p["cuda_graphs"] or not _artifact_snapshot_matches(p["preflight_artifacts"]):
        raise ValueError("Compiled protocol or preflight fingerprint mismatch")
    earlier, _ = previous_runs()
    config, stop = Config(**p["config"]), StopConfig(**p["stopping"])
    rows = read_results(directory / "runs.jsonl")
    expected = {f"{a}:{m['name']}:{s}" for a in p["attention"] for m in p["methods"] for s in p["seeds"]}
    actual = [r["id"] for r in rows]
    if len(actual) != len(set(actual)) or not set(actual) <= expected or complete and set(actual) != expected:
        raise ValueError("Unexpected, duplicate, or incomplete run identifiers")
    data = TextData.load(_data_file, "cuda")
    if data.sha256 != p["data_sha256"]:
        raise ValueError("Dataset fingerprint mismatch")
    checks = []
    rates = selected_rates()
    for row in rows:
        check_initial(row, earlier)
        if row["protocol_sha256"] != fingerprint or row["lr"] != rates[row["attention"]][row["method"]]:
            raise ValueError("Unrecorded protocol or changed selected rate")
        starts = training_starts(len(data.train), config.max_seq_len, p["batch"], stop.max_steps, row["seed"])
        if hashlib.sha256(starts.numpy().tobytes()).hexdigest() != row["batch_plan_sha256"]:
            raise ValueError("Unpaired full minibatch stream")
        if hashlib.sha256(starts[:row["actual_steps"]].numpy().tobytes()).hexdigest() != row["batch_sha256"]:
            raise ValueError("Logged minibatch prefix mismatch")
        replay = EarlyStopper(stop)
        last_decision = None
        for curve in row["curves"]:
            if last_decision is not None and last_decision.should_stop:
                raise ValueError("Training continued past its stopping decision")
            value = curve["validation_loss"] if curve["validation_loss"] is not None else math.nan
            last_decision = replay.observe(curve["step"], value)
            if (last_decision.new_best != curve["new_best"] or replay.bad_checks != curve["bad_checks"]
                    or replay.divergent_checks != curve["divergent_checks"] or last_decision.reason != curve["stop_reason"]):
                raise ValueError("Controller replay differs from the logged decision")
        if row["stop_reason"] in {"patience", "validation_divergence", "max_steps", "nonfinite_validation"}:
            if last_decision.reason != row["stop_reason"] or not last_decision.should_stop:
                raise ValueError("Run has no matching stopping decision")
        if row["status"] == "failed":
            continue
        if row["best_step"] != choose_best_step(row["curves"]) or row["test_evaluations"] != 1:
            raise ValueError("Checkpoint selected using something other than validation")
        checkpoint = _artifact_path(row["checkpoint"])
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != row["checkpoint_sha256"]:
            raise ValueError("Best checkpoint fingerprint changed")
        checks.append(check_checkpoint(row, config, data, p["batch"]))
    stored = json.loads((directory / "summary.json").read_text())["methods"]
    if stored != summarize(rows, p["seeds"]):
        raise ValueError("Summary differs from raw measurements")
    result = {"runs": len(rows), "expected_runs": len(expected), "complete": set(actual) == expected,
              "failed": sum(r["status"] == "failed" for r in rows),
              "recovered": sum(r["status"] == "recovered" for r in rows),
              "capped": sum(r["stop_reason"] == "max_steps" for r in rows),
              "stopping_replayed": True, "initialization_and_minibatch_streams_paired": True,
              "test_unused_for_stopping": True, "previous_experiments_unchanged": True,
              "raw_sha256": hashlib.sha256((directory / "runs.jsonl").read_bytes()).hexdigest(),
              "compiled_cuda_checkpoint_checks": True, "checkpoint_checks": checks}
    write_json(directory / "validation.json", result)
    return result


def check_checkpoint(row, config, data, batch):
    """Reload into the same compiled backend, retaining Fisher's hook setup."""
    torch.compiler.reset()
    payload = torch.load(_artifact_path(row["checkpoint"]), map_location="cpu", weights_only=True)
    model = compile_model(make_model(config, row["attention"], row["seed"], "cuda"),
                          row["attention"], row["method"])
    optimizer = make_optimizer(row["method"], model, row["lr"], row["seed"])
    try:
        model.load_state_dict(payload["model"])
        if state_hash(model) != row["best_model_sha256"]:
            raise ValueError("Checkpoint model does not match selected weights")
        val = evaluate(model, data.validation, config.max_seq_len, batch)
        test = evaluate(model, data.test, config.max_seq_len, batch)
        if abs(val - row["validation_loss"]) > 1e-9 or abs(test - row["test_loss"]) > 1e-9:
            raise ValueError("Compiled checkpoint fails held-out loss re-evaluation")
        return {"id": row["id"], "best_step": row["best_step"],
                "validation_loss": val, "test_loss": test, **graph_counts()}
    finally:
        optimizer.close()
        del model, optimizer, payload
        torch.compiler.reset()
        gc.collect()
        torch.cuda.empty_cache()
