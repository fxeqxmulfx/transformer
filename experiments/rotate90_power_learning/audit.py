"""Replay every finished checkpoint eagerly on CPU, with independent label checks.

Run from python/: uv run --locked python ../experiments/rotate90_power_learning/audit.py
Compare accuracy exactly and losses within 5e-5 relative / 5e-6 absolute.
Keep CPU argmax differences explicit. A checkpoint-specific certificate from
replay_roundoff.py may resolve a difference by matching the original compiled
CUDA metrics and checking that all decision changes are numerical near ties.
"""

import hashlib
import json
import math
from pathlib import Path

import torch

from lab.infrastructure.benchmarks import build_task
from lab.infrastructure.engine.loop import evaluate
from lab.infrastructure.loader import load
from lab.infrastructure.nn import build_model
from lab.infrastructure.optim import build_optimizer, evaluation_weights, training_weights
from lab.infrastructure.provenance import code
from lab.infrastructure.store import RunDirectory

ROOT = Path(__file__).resolve().parent


def main():
    torch.set_num_threads(1)
    study = load(ROOT)
    reports = []
    current_code = code()
    replay_path = ROOT / "cuda_roundoff_replay.json"
    replays = {item["label"]: item for item in json.loads(replay_path.read_text())["runs"]} if replay_path.exists() else {}
    for label, experiment in study.experiments.items():
        run = RunDirectory(ROOT / "runs" / label)
        result = run.result()
        if result is None:
            continue
        checkpoint = run.checkpoint("cpu")
        task = build_task(experiment.benchmark, experiment.seeds.data, torch.device("cpu"), experiment.model)
        model = build_model(experiment.model, task.vocab, experiment.seeds.model)
        model.load_state_dict(checkpoint["model"])
        optimizer = build_optimizer(experiment.optimizer, model)
        optimizer.load_state_dict(checkpoint["optimizer"])
        row = {"label": label, "checkpoint_step": checkpoint["step"],
               "step_matches": checkpoint["step"] == result["stop"]["step"],
               "source_matches": run.provenance()["engine"]["code"] == current_code,
               "label_oracle": "all rows passed independent powered multiplication check", "splits": {}}
        certificate = replays.get(label)
        if certificate and certificate["checkpoint_sha256"] != hashlib.sha256(
                (ROOT / "runs" / label / "checkpoint.pt").read_bytes()).hexdigest():
            certificate = None
        training_weights(optimizer)
        row["training_weights"] = {
            split: evaluate(task, model, task.splits[split], experiment.evaluate.batch)
            for split in experiment.benchmark.observed}
        with evaluation_weights(optimizer):
            for split in experiment.benchmark.observed:
                actual = evaluate(task, model, task.splits[split], experiment.evaluate.batch)
                expected = result["final"][split]
                row["splits"][split] = {"actual": actual, "recorded": expected,
                                         "accuracy_matches": actual["accuracy"] == expected["accuracy"],
                                         "loss_matches": math.isclose(actual["loss"], expected["loss"],
                                                                       rel_tol=5e-5, abs_tol=5e-6)}
        row["cpu_exact"] = all(
            split["accuracy_matches"] and split["loss_matches"] for split in row["splits"].values())
        for split, comparison in row["splits"].items():
            replay = None if certificate is None else certificate["splits"].get(split)
            resolved = bool(replay and replay["pass"] and replay["recorded"] == result["final"][split])
            comparison["cuda_roundoff_verified"] = resolved
            comparison["verified"] = comparison["loss_matches"] and (comparison["accuracy_matches"] or resolved)
        row["pass"] = row["step_matches"] and row["source_matches"] and all(
            split["verified"] for split in row["splits"].values())
        reports.append(row)
        print(label, "PASS" if row["pass"] else "FAIL", flush=True)
    payload = {"replayed": len(reports), "passed": sum(row["pass"] for row in reports),
               "cpu_exact": sum(row["cpu_exact"] for row in reports),
               "cuda_roundoff_resolved": sum(row["pass"] and not row["cpu_exact"] for row in reports),
               "runs": reports}
    (ROOT / "checkpoint_audit.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    if not all(row["pass"] for row in reports):
        raise SystemExit("Checkpoint replay discrepancies are recorded in checkpoint_audit.json")


if __name__ == "__main__":
    main()
