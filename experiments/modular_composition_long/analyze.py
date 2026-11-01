"""Export all bounded-continuation outcomes and replay saved final states.

Never select a checkpoint by heldout performance. A timeout's saved endpoint
is explicitly unfinished; observations beyond it are retained separately.
Run only after the supervisor records the end of the shared wall-time cap.
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
ROOT = HERE.parents[1]


def transitions(history):
    previous = None
    records = []
    for row in history:
        if "heldout" not in row:
            continue
        success = row["heldout"]["accuracy"] >= .99
        if previous is None or previous != success:
            records.append({"step": row["step"], "epochs_seen": row["epochs_seen"],
                            "accuracy": row["heldout"]["accuracy"],
                            "state": "at_least99_percent" if success else "below99_percent"})
        previous = success
    return records


def analyze(case, experiment):
    run = RunDirectory(HERE / "runs" / case["label"])
    source_path = ROOT / experiment.checkpoint.source
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == experiment.checkpoint.sha256
    result = {**case, "specification": describe(experiment), "source_sha256": experiment.checkpoint.sha256}
    checkpoint = run.checkpoint("cpu")
    if checkpoint is None:
        return result | {"replay_status": "no_checkpoint"}
    assert run.description() == describe(experiment)
    provenance = run.provenance()
    assert provenance["python"].startswith("3.15.")
    assert provenance["engine"]["torch"].startswith("2.15.")
    assert not provenance["commit"]["lab_modified"]
    source = torch.load(source_path, map_location="cpu", weights_only=True)
    initial = torch.load(run.path / "checkpoints" / "step-00002000.pt", map_location="cpu", weights_only=True)
    # Only wall/diagnostic time can change during compilation and initial inspection.
    for key in source:
        if key not in ("wall_seconds", "diagnostic_seconds"):
            torch.testing.assert_close(initial[key], source[key], rtol=0, atol=0)
    assert checkpoint["optimizer"]["param_groups"][0]["step"] == checkpoint["step"]
    history = run.records("history")
    early = RunDirectory(source_path.parent).records("history")
    combined = [r for r in early if r["step"] < 2000] + history
    final = run.result()
    if case["status"] == "completed":
        assert checkpoint["step"] == experiment.budget.updates
        assert final["stop"] == {"step": experiment.budget.updates, "reason": "budget"}
    task = build_task(experiment.benchmark, experiment.seeds.data, torch.device("cpu"))
    model = build_model(experiment.model, task.vocab, experiment.seeds.model)
    model.load_state_dict(checkpoint["model"])
    stepper = EagerStepper(experiment, task, model, None)
    stepper.load_state_dict(checkpoint["optimizer"])
    recorded = next((r for r in reversed(history) if r["step"] == checkpoint["step"]), None)
    replays = {}
    for split in (*experiment.benchmark.observed,
                  *(experiment.benchmark.last if case["status"] == "completed" else ())):
        actual = stepper.evaluate(split)
        expected = (final["last"][split] if split in experiment.benchmark.last else
                    recorded[split] if recorded else None)
        replay = {"actual": actual, "recorded": expected}
        if expected is not None:
            replay.update(loss_absolute_difference=abs(actual["loss"] - expected["loss"]),
                          correct_count_difference=round((actual["accuracy"] - expected["accuracy"]) * actual["examples"]),
                          loss_within_declared_tolerance=math.isclose(actual["loss"], expected["loss"], rel_tol=3e-4, abs_tol=3e-4),
                          same_99_percent_classification=(actual["accuracy"] >= .99) == (expected["accuracy"] >= .99))
        replays[split] = replay
    curve = [{k: row[k] for k in ("step", "epochs_seen", "training_seconds", "wall_seconds", "train", "heldout")
              if k in row} for row in combined]
    result.update(checkpoint_step=checkpoint["step"], examples_seen=checkpoint["examples_seen"],
                  epochs_seen=checkpoint["examples_seen"] / task.train_count,
                  source_initial_state_bitwise_equal=True,
                  checkpoint_sha256=hashlib.sha256((run.path / "checkpoint.pt").read_bytes()).hexdigest(),
                  retained_checkpoints={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                        for p in sorted((run.path / "checkpoints").glob("*.pt"))},
                  replay_status="eager_cpu", replay=replays, learning_curve=curve,
                  transitions=transitions(combined), task=task.summary(),
                  provenance=provenance, recorded_result=final,
                  last_observed_step=history[-1]["step"],
                  observation_after_last_saved_state=history[-1]["step"] > checkpoint["step"])
    return result


def main():
    torch.set_num_threads(1)
    phase = json.loads((HERE / "runs" / "phase.json").read_text())
    assert "ended_utc" in phase, "Wait for the single bounded phase to finish"
    assert phase["wall_seconds"] <= phase["wall_budget_seconds"]
    assert hashlib.sha256((HERE / "hypotheses.json").read_bytes()).hexdigest() == phase["protocol_sha256"]
    assert hashlib.sha256((HERE / "experiment.py").read_bytes()).hexdigest() == phase["experiment_sha256"]
    study = load(HERE)
    assert [c["label"] for c in phase["cases"]] == list(study.experiments)
    results = []
    for case in phase["cases"]:
        record = analyze(case, study.experiments[case["label"]])
        results.append(record)
        print(json.dumps({"label": case["label"], "status": case["status"],
                          "checkpoint_step": record.get("checkpoint_step"),
                          "transitions": record.get("transitions")}), flush=True)
    report = {"phase": phase, "cases": results,
              "scope": "All declared cases; no checkpoint selection. CPU replay discrepancies retained for compiled follow-up. Timeouts are unfinished.",
              "full_state_scope": "Initial model, complete optimizer, sampler and counters compared bitwise with each frozen source; only inspection timing excluded."}
    (HERE / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
