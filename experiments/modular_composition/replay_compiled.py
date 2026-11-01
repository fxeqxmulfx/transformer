"""Replay every eager-discrepant endpoint using the original compiled path.

Compilation is warmed before restoring the exact saved query and optimizer.
The checkpoint's extra evaluation roundtrip can still introduce FP32 error;
report it explicitly instead of widening tolerances or changing a checkpoint.
"""

import copy
import hashlib
import json
import math
from pathlib import Path

import torch

from lab.infrastructure.benchmarks import build_task
from lab.infrastructure.engine.compiled import CompiledStepper
from lab.infrastructure.loader import load
from lab.infrastructure.nn import build_model
from lab.infrastructure.store import RunDirectory

HERE = Path(__file__).parent


def main():
    torch.set_num_threads(1)
    report = json.loads((HERE / "results.json").read_text())
    study = load(HERE)
    for row in report["runs"]:
        flagged = [s for s, r in row["replay"].items()
                   if not r["loss_within_declared_tolerance"] or r["correct_count_difference"]]
        if not flagged:
            continue
        experiment = study.experiments[row["label"]]
        run = RunDirectory(HERE / "runs" / row["label"])
        assert hashlib.sha256((run.path / "checkpoint.pt").read_bytes()).hexdigest() == row["checkpoint_sha256"]
        checkpoint = run.checkpoint("cuda")
        task = build_task(experiment.benchmark, experiment.seeds.data, torch.device("cuda"))
        model = build_model(experiment.model, task.vocab, experiment.seeds.model).cuda()
        model.load_state_dict(checkpoint["model"])
        stepper = CompiledStepper(experiment, task, model, None)
        stepper.load_state_dict(copy.deepcopy(checkpoint["optimizer"]))
        for split in flagged:
            stepper.evaluate(split)
        model.load_state_dict(checkpoint["model"])
        stepper.load_state_dict(copy.deepcopy(checkpoint["optimizer"]))
        for split in (*experiment.benchmark.observed, *experiment.benchmark.last):
            actual = stepper.evaluate(split)
            recorded = row["replay"][split]["recorded"]
            result = {"actual": actual, "loss_absolute_difference": abs(actual["loss"] - recorded["loss"]),
                      "correct_count_difference": round((actual["accuracy"] - recorded["accuracy"]) * actual["examples"]),
                      "loss_within_declared_tolerance": math.isclose(actual["loss"], recorded["loss"], rel_tol=3e-4, abs_tol=3e-4),
                      "same_99_percent_classification": (actual["accuracy"] >= .99) == (recorded["accuracy"] >= .99)}
            row["replay"][split]["compiled_followup"] = result
            print(json.dumps({"label": row["label"], "split": split, **result}), flush=True)
        torch.cuda.empty_cache()
    report["replay_scope"] = ("All48 final checkpoints independently replayed on eager CUDA; every discrepant case also replayed on the original compiled path. Original3e-4 absolute/relative loss tolerance and exact answer counts retained. Saving training queries after evaluation can add an FP32 mode-roundtrip difference on replay; all differences are reported.")
    (HERE / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
