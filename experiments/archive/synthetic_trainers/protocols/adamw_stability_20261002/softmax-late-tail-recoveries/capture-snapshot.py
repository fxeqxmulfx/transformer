"""Preserve two actual late softmax failures, recoveries and complete frequency bins.

Read-only partial witness: the unchanged full 300000-update budget continues.
Observed metric A/A checks cannot establish unseen dynamics or independent repeats.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

from experiments.synthetic_trainers.attention_layout import digest
from experiments.synthetic_trainers.persistence import PersistenceConfig, assess
from experiments.synthetic_trainers.stability_integrity import validate_observation
from experiments.synthetic_trainers.stability_recovery import describe
from experiments.synthetic_trainers.stability_recovery_series import timeline
from experiments.synthetic_trainers.stability_report import write_json

PROTOCOL = Path("experiments/synthetic_trainers/protocols/adamw_stability_20261002")
THROUGH = 280000
BOUNDARIES = (274000, 274250, 275500, 275750)


def rows(path):
    result = []
    for line in Path(path).read_text().splitlines():
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # Exact prefix/triplet counts reject missing captured records.
    return result


def capture(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Late scientific recovery witnesses require a fresh destination")
    plan = json.loads((PROTOCOL / "attention-pair-plan.json").read_text())
    raw = Path(plan["output_directory"]); case = raw / "adamw-softmax"
    assert json.loads((raw / "plan.json").read_text()) == plan
    recipe = next(r for r in plan["recipes"] if r["name"] == "adamw-softmax")
    native = json.loads((case / "plan.json").read_text())
    assert native["config"] == recipe["config"] and native["config"]["steps"] == 300000
    assert native["source_hashes"] == plan["training_source_hashes"] and native["corpus"] == plan["corpus"]
    for field in ("source_hashes", "training_source_hashes", "Lean_specification_hashes", "papers"):
        assert all(digest(p) == h for p, h in plan[field].items())
    assert digest(PROTOCOL / "sparsemax-preparation/validation.json") == plan["Lean_audit_validation_sha256"]
    history = [r for r in rows(case / "history.jsonl") if r["step"] <= THROUGH]
    assert [r["step"] for r in history] == list(range(0, THROUGH + 1, 250))
    selected = {step + offset for step in BOUNDARIES for offset in (-1, 0, 1)}
    probes = [r for r in rows(case / "probes.jsonl") if r["step"] in selected]
    assert len(probes) == 8 and {r["step"] for r in probes} == selected - set(BOUNDARIES)
    diagnostics = [r for r in rows(case / "diagnostics.jsonl") if r["step"] in selected]
    assert [r["step"] for r in diagnostics] == sorted(selected)
    gradients = []
    with (case / "gradients.jsonl").open() as stream:
        for line in stream:
            try:
                point = json.loads(line)
            except json.JSONDecodeError:
                continue
            if point["step"] in selected:
                gradients.append(point)
            if point["step"] > max(selected):
                break
    assert [r["step"] for r in gradients] == sorted(selected)
    for point in [*history, *probes]:
        validate_observation(point, plan, recipe["config"])
    report = {"plan": {**native, "status": "running"}, "history": history,
              "completed_steps": THROUGH, "final": history[-1]}
    criterion = PersistenceConfig(**plan["criterion"])
    assessment = assess(report, criterion)
    recovery = describe(report, criterion, window_steps=10000)
    series = timeline(report, criterion, window_steps=10000)
    assert not assessment["complete_canonical_history"] and assessment["tail_observations"] == 121
    assert assessment["tail_failures"] == [274000, 275500]
    episodes = recovery["metrics"]["joint"]["episodes_after_long_onset"]
    assert len(episodes) == 6
    assert [(e["first_failure"], e["first_recovery"], e["sampled_steps_until_first_recovery"])
            for e in episodes[-2:]] == [(274000, 274250, 250), (275500, 275750, 250)]
    reference_key = "runs/adamw-mod193-fraction25-lr0003-budget300k/measurements.json"
    reference_path = raw / "reference" / reference_key
    assert digest(reference_path) == plan["reference_hashes"][reference_key]
    previous = [r for r in json.loads(reference_path.read_text())["history"] if r["step"] <= THROUGH]
    strip = lambda p: {k: v for k, v in p.items()
                      if k not in ("training_seconds", "wall_seconds", "diagnostic_seconds")}
    assert [strip(r) for r in history] == [strip(r) for r in previous]
    output.mkdir(parents=True)
    shutil.copyfile(__file__, output / "capture-snapshot.py")
    write_json(output / "frozen-root-plan.json", plan)
    write_json(output / "captured-native-case-plan.json", native)
    (output / "canonical-prefix.jsonl").write_text("".join(json.dumps(r, allow_nan=False) + "\n" for r in history))
    (output / "reference-non-time-prefix.jsonl").write_text("".join(json.dumps(strip(r), allow_nan=False) + "\n" for r in previous))
    write_json(output / "boundary-triplets.json", {"canonical": [r for r in history if r["step"] in BOUNDARIES],
        "neighbors": probes, "gradients": gradients, "tensor_diagnostics": diagnostics})
    write_json(output / "assessment.json", assessment)
    write_json(output / "recovery-metrics.json", recovery)
    write_json(output / "recovery-series.json", series)
    assert "torch" not in sys.modules
    write_json(output / "validation.json", {"captured_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "partial_actual_softmax_late_tail_failures_and_recoveries; full_budget_continues",
        "through_step": THROUGH, "canonical_observations": len(history), "final_window_observations": 121,
        "full_budget_complete": False, "frozen_final_window": [250000, 300000],
        "cannot_meet_frozen_all_observations_final_tail": True,
        "observed_non_time_metric_A_A_matches_frozen_reference_prefix": True,
        "parameter_tensor_or_unobserved_trajectory_identity_claimed": False,
        "all_43_Python_17_native_9_Lean_2_papers_and_audit_unchanged": True,
        "optimizer_updates_performed": 0, "Torch_imported": False,
        "scientific_confirmation_schedule_or_architecture_selected": False, "goal_complete": False})
    write_json(output / "artifact-hashes.json", {"files": {str(p.relative_to(output)): digest(p)
        for p in sorted(output.rglob("*")) if p.is_file() and p.name != "artifact-hashes.json"}})
    return assessment, episodes


if __name__ == "__main__":
    result, episodes = capture(sys.argv[1])
    print(json.dumps({"through_step": THROUGH, "canonical_observations": 1121,
                      "tail_failures": result["tail_failures"], "late_episodes": episodes[-2:]}))
