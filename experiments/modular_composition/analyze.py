"""Export every declared outcome and audit the saved averaged-weight models.

Run with uv run --locked inside python/. Checkpoints are evaluated eagerly
on CUDA as an independent execution path; small loss rounding differences
are reported, and answer counts must match the recorded compiled results.
No model or checkpoint is selected by this report.
"""

import hashlib
import json
import math
from pathlib import Path

import torch

from lab.domain.spec import describe
from lab.infrastructure.benchmarks import build_task
from lab.infrastructure.engine.eager import EagerStepper
from lab.infrastructure.loader import load
from lab.infrastructure.nn import build_model
from lab.infrastructure.store import RunDirectory

HERE = Path(__file__).parent


def analyze(label, experiment):
    run = RunDirectory(HERE / "runs" / label)
    result = run.result()
    if result is None:
        return {"label": label, "status": "missing", "specification": describe(experiment)}
    assert run.description() == describe(experiment)
    provenance = run.provenance()
    assert provenance["python"].startswith("3.15.")
    assert provenance["engine"]["torch"].startswith("2.15.")
    assert not provenance["commit"]["lab_modified"]
    assert result["stop"] == {"step": experiment.budget.updates, "reason": "budget"}
    device = torch.device("cuda")
    task = build_task(experiment.benchmark, experiment.seeds.data, device)
    model = build_model(experiment.model, task.vocab, experiment.seeds.model).to(device)
    checkpoint = run.checkpoint(device)
    assert checkpoint["step"] == experiment.budget.updates
    model.load_state_dict(checkpoint["model"])
    stepper = EagerStepper(experiment, task, model, None)
    stepper.load_state_dict(checkpoint["optimizer"])
    assert result["final"]["task"] == task.summary()
    expected_parameters = task.vocab * experiment.model.width + experiment.model.depth * (
        4 * experiment.model.width**2 + 3 * experiment.model.width * experiment.model.hidden)
    assert sum(p.numel() for p in model.parameters()) == expected_parameters
    replays = {}
    for split in (*experiment.benchmark.observed, *experiment.benchmark.last):
        recorded = result["final"][split] if split in experiment.benchmark.observed else result["last"][split]
        actual = stepper.evaluate(split)
        error = abs(actual["loss"] - recorded["loss"])
        count_error = round((actual["accuracy"] - recorded["accuracy"]) * actual["examples"])
        assert math.isclose(actual["loss"], recorded["loss"], rel_tol=3e-4, abs_tol=3e-4), (label, split, error)
        assert count_error == 0, (label, split, count_error)
        replays[split] = {"loss_absolute_difference": error, "correct_count_difference": count_error}
    history = run.records("history")
    record = {"label": label, "status": "completed", "specification": describe(experiment),
              "parameters": expected_parameters, "head_dimension": experiment.model.width // experiment.model.heads,
              "training_seconds": result["training_seconds"], "wall_seconds": result["wall_seconds"],
              "peak_cuda_allocated_bytes": result["peak_cuda_allocated_bytes"],
              "examples_seen": checkpoint["examples_seen"], "task": task.summary(), "replay": replays,
              "final": {name: result["final"][name] for name in experiment.benchmark.observed},
              "best_heldout_accuracy": max((r["heldout"]["accuracy"] for r in history if "heldout" in r), default=None),
              "longer": result.get("last", {}).get("longer"),
              "checkpoint_sha256": hashlib.sha256((run.path / "checkpoint.pt").read_bytes()).hexdigest(),
              "source_commit": provenance["commit"]["head"], "python": provenance["python"],
              "torch": provenance["engine"]["torch"], "gpu": provenance["engine"]["device"]}
    return record


def main():
    torch.set_num_threads(1)
    study = load(HERE)
    results = []
    for label, experiment in study.experiments.items():
        record = analyze(label, experiment)
        results.append(record)
        print(json.dumps({"label": label, "status": record["status"], "final": record.get("final"),
                          "longer": record.get("longer")}), flush=True)
        torch.cuda.empty_cache()
    report = {"protocol_sha256": hashlib.sha256((HERE / "hypotheses.json").read_bytes()).hexdigest(),
              "completed": sum(row["status"] == "completed" for row in results), "declared": len(results),
              "runs": results}
    (HERE / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    assert report["completed"] == report["declared"]


if __name__ == "__main__":
    main()
