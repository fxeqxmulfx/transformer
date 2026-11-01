"""Retain CPU discrepancies and replay flagged long endpoints under compilation.

Uses the frozen source model, never the newer diagnostic XSA correction.
Original tolerances and exact answer counts are retained even when they fail.
"""

import copy
import gc
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
    for row in report["cases"]:
        replays = row.get("replay", {})
        flagged = [s for s, r in replays.items() if r["recorded"] is not None and
                   (not r["loss_within_declared_tolerance"] or r["correct_count_difference"])]
        if not flagged:
            continue
        torch._dynamo.reset()
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
        for split, replay in replays.items():
            recorded = replay["recorded"]
            if recorded is None:
                continue
            actual = stepper.evaluate(split)
            result = {
                "actual": actual,
                "loss_absolute_difference": abs(actual["loss"] - recorded["loss"]),
                "correct_count_difference": round((actual["accuracy"] - recorded["accuracy"]) * actual["examples"]),
                "loss_within_declared_tolerance": math.isclose(actual["loss"], recorded["loss"], rel_tol=3e-4, abs_tol=3e-4),
                "same_99_percent_classification": (actual["accuracy"] >= .99) == (recorded["accuracy"] >= .99),
            }
            replay["compiled_followup"] = result
            print(json.dumps({"label": row["label"], "split": split, **result}), flush=True)
        del stepper, model, task, checkpoint
        gc.collect()
        torch.cuda.empty_cache()
    report["replay_scope"] = (
        "All saved endpoints replayed on CPU; every discrepant endpoint also replayed "
        "using original compiled CUDA. Original3e-4 absolute/relative loss tolerance "
        "and exact counts retained. Saving training queries after evaluation can add "
        "an FP32 mode-roundtrip difference; this is not isolated as the sole cause.")
    (HERE / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
