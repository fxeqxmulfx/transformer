"""Preserve the scheduled constant control phase and first sampled recovery.

This is a partial scientific prefix, not final persistence or an independent
repeat. Original 300000-update budgets, sources and all gates remain intact.
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
from experiments.synthetic_trainers.stability_report import write_json

PROTOCOL = Path("experiments/synthetic_trainers/protocols/adamw_stability_20261002")
THROUGH = 105750
BOUNDARIES = (100500, 105250, 105500, 105750)


def rows(path):
    result = []
    for line in Path(path).read_text().splitlines():
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # Exact prefix/triplet counts below reject missing captured records.
    return result


def capture(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError("A scientific prefix witness requires a fresh destination")
    plan = json.loads((PROTOCOL / "scheduled-pair-plan.json").read_text())
    raw = Path(plan["output_directory"]); case = raw / "adamw-constant"
    assert json.loads((raw / "plan.json").read_text()) == plan
    recipe = next(r for r in plan["recipes"] if r["name"] == "adamw-constant")
    native = json.loads((case / "plan.json").read_text())
    assert native["config"] == recipe["config"] and native["source_hashes"] == plan["training_source_hashes"]
    assert native["config"]["steps"] == 300000 and native["corpus"] == plan["corpus"]
    assert native["config"]["learning_rate_schedule"] == "constant" and native["parameters"] == 436104
    for field in ("source_hashes", "training_source_hashes", "Lean_specification_hashes", "papers"):
        assert all(digest(p) == h for p, h in plan[field].items())
    assert digest(PROTOCOL / "sparsemax-preparation/validation.json") == plan["Lean_audit_validation_sha256"]
    history = [r for r in rows(case / "history.jsonl") if r["step"] <= THROUGH]
    assert [r["step"] for r in history] == list(range(0, THROUGH + 1, 250))
    selected = {s + offset for s in BOUNDARIES for offset in (-1, 0, 1)}
    probes = [r for r in rows(case / "probes.jsonl") if r["step"] in selected]
    assert {r["step"] for r in probes} == selected - set(BOUNDARIES) and len(probes) == 8
    diagnostics = [r for r in rows(case / "diagnostics.jsonl") if r["step"] in selected]
    assert [r["step"] for r in diagnostics] == sorted(selected)
    gradients = []
    with (case / "gradients.jsonl").open() as stream:
        for line in stream:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record["step"] in selected:
                gradients.append(record)
            if record["step"] > max(selected):
                break
    assert [r["step"] for r in gradients] == sorted(selected)
    for point in [*history, *probes]:
        validate_observation(point, plan, recipe["config"])
    report = {"plan": {**native, "status": "running"}, "history": history,
              "completed_steps": THROUGH, "final": history[-1]}
    criterion = PersistenceConfig(**plan["criterion"])
    assessment = assess(report, criterion); recovery = describe(report, criterion, window_steps=10000)
    assert assessment["plateau"] == {"start": 4500, "end": 27250, "observations": 92}
    assert assessment["long_confirmation"]["onset"] == 95750 and assessment["long_confirmation"]["confirmed"] == 100500
    assert not assessment["complete_canonical_history"] and assessment["tail_observations"] == 0
    episode = recovery["metrics"]["joint"]["episodes_after_long_onset"]
    assert len(episode) == 1 and episode[0]["first_failure"] == 105250 and episode[0]["last_failure"] == 105500
    assert episode[0]["first_recovery"] == 105750 and episode[0]["sampled_steps_until_first_recovery"] == 500
    reference_key = "runs/adamw-softmax/measurements.json"
    reference_path = raw / "reference" / reference_key
    assert digest(reference_path) == plan["reference_hashes"][reference_key]
    reference = json.loads(reference_path.read_text())
    previous = [p for p in reference["history"] if p["step"] <= THROUGH]
    strip = lambda p: {k: v for k, v in p.items() if k not in ("training_seconds", "wall_seconds", "diagnostic_seconds")}
    assert [strip(p) for p in history] == [strip(p) for p in previous]
    output.mkdir(parents=True)
    shutil.copyfile(__file__, output / "capture-snapshot.py")
    write_json(output / "frozen-root-plan.json", plan)
    write_json(output / "captured-native-case-plan.json", native)
    (output / "canonical-prefix.jsonl").write_text("".join(json.dumps(p, allow_nan=False) + "\n" for p in history))
    (output / "reference-non-time-prefix.jsonl").write_text("".join(json.dumps(strip(p), allow_nan=False) + "\n" for p in previous))
    write_json(output / "boundary-triplets.json", {"canonical": [p for p in history if p["step"] in BOUNDARIES],
        "neighbors": probes, "gradients": gradients, "tensor_diagnostics": diagnostics})
    write_json(output / "assessment.json", assessment); write_json(output / "recovery-metrics.json", recovery)
    assert "torch" not in sys.modules
    write_json(output / "validation.json", {"captured_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "partial_scheduled_constant_phase_and_first_sampled_recovery; no_final_persistence_claim",
        "through_step": THROUGH, "canonical_observations": len(history), "scientific_run_complete": False,
        "frozen_final_window": [250000, 300000], "final_window_observations": 0,
        "A_A_all_424_observed_non_time_canonical_metrics_exactly_match_reference": True,
        "A_A_claims_parameter_tensor_or_unobserved_trajectory_identity": False,
        "all_55_Python_17_native_9_Lean_2_papers_and_audit_unchanged": True,
        "optimizer_updates_performed": 0, "Torch_imported": False,
        "first_sampled_episode_count": 1, "full_pair_and_scientific_goal_complete": False})
    write_json(output / "artifact-hashes.json", {"files": {str(p.relative_to(output)): digest(p)
        for p in sorted(output.rglob("*")) if p.is_file() and p.name != "artifact-hashes.json"}})
    return assessment, episode


if __name__ == "__main__":
    assessment, episodes = capture(sys.argv[1])
    print(json.dumps({"through_step": THROUGH, "long_confirmation": assessment["long_confirmation"],
                      "first_sampled_recovery": episodes[0], "final_window_observations": 0, "Torch_imported": False}))
