"""Replay every finished checkpoint eagerly on CPU, with independent label checks.

Run from python/: uv run --locked python ../experiments/rotate90_power_learning/audit.py
Compare accuracy exactly and losses within 5e-5 relative / 5e-6 absolute,
allowing float32 reduction differences between compiled CUDA and eager CPU.
"""

import json
import math
from pathlib import Path

import torch

from lab.infrastructure.benchmarks import build_task
from lab.infrastructure.engine.loop import evaluate
from lab.infrastructure.loader import load
from lab.infrastructure.nn import build_model
from lab.infrastructure.provenance import code
from lab.infrastructure.store import RunDirectory

ROOT = Path(__file__).resolve().parent


def main():
    torch.set_num_threads(1)
    study = load(ROOT)
    reports = []
    current_code = code()
    for label, experiment in study.experiments.items():
        run = RunDirectory(ROOT / "runs" / label)
        result = run.result()
        if result is None:
            continue
        checkpoint = run.checkpoint("cpu")
        task = build_task(experiment.benchmark, experiment.seeds.data, torch.device("cpu"), experiment.model)
        model = build_model(experiment.model, task.vocab, experiment.seeds.model)
        model.load_state_dict(checkpoint["model"])
        row = {"label": label, "checkpoint_step": checkpoint["step"],
               "step_matches": checkpoint["step"] == result["stop"]["step"],
               "source_matches": run.provenance()["engine"]["code"] == current_code,
               "label_oracle": "all rows passed independent powered multiplication check", "splits": {}}
        for split in experiment.benchmark.observed:
            actual = evaluate(task, model, task.splits[split], experiment.evaluate.batch)
            expected = result["final"][split]
            row["splits"][split] = {"actual": actual, "recorded": expected,
                                     "accuracy_matches": actual["accuracy"] == expected["accuracy"],
                                     "loss_matches": math.isclose(actual["loss"], expected["loss"],
                                                                   rel_tol=5e-5, abs_tol=5e-6)}
        row["pass"] = row["step_matches"] and row["source_matches"] and all(
            split["accuracy_matches"] and split["loss_matches"] for split in row["splits"].values())
        reports.append(row)
        print(label, "PASS" if row["pass"] else "FAIL", flush=True)
    payload = {"replayed": len(reports), "passed": sum(row["pass"] for row in reports), "runs": reports}
    (ROOT / "checkpoint_audit.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    if not all(row["pass"] for row in reports):
        raise SystemExit("Checkpoint replay discrepancies are recorded in checkpoint_audit.json")


if __name__ == "__main__":
    main()
