"""Freeze a compact, fingerprinted snapshot of the six offline observations.

From python/: uv run --locked python ../experiments/grokking_internals/summarize.py.
Threshold crossings are retrospective evaluation at the recorded cadence,
not newly fitted grokking alarms. Sparse endpoint intervals stay explicit.
"""

import hashlib
import json
from pathlib import Path

from lab.infrastructure.loader import load

HERE = Path(__file__).resolve().parent
STUDIES = (HERE, HERE.parent / "grokking_progress")


def read(path):
    data = path.read_bytes()
    lines = data.splitlines()
    if data and not data.endswith(b"\n"):
        lines = lines[:-1]
    prefix = b"\n".join(lines) + (b"\n" if lines else b"")
    return [json.loads(line) for line in lines], hashlib.sha256(prefix).hexdigest()


def compact(row):
    result = {key: row[key] for key in ("step", "checkpoint_sha256", "checkpoint_origin", "observation_device",
                                       "training_device", "observation_seconds", "output", "gradients", "neurons",
                                       "ablations", "functional_update")}
    result["features"] = {}
    for name, feature in row["features"].items():
        result["features"][name] = {**feature, **{
            key: {k: value for k, value in feature[key].items() if k != "eigenvalue_energy_fractions"}
            for key in ("train_spectrum", "heldout_spectrum")}}
    return result


def first(rows, predicate):
    return next((row["step"] for row in rows if predicate(row)), None)


def snapshot():
    report = {"scope": "six_offline_measurements_on_current_retained_weights; no_universal_prediction_claim",
              "study_choices": {"probe_ridge": .001, "gradient_batches": 8, "gradient_batch_size": 32,
                                "subspace_rank": 8, "threshold_crossings": .99}, "runs": {}}
    for study in STUDIES:
        for label, experiment in load(study).select([]):
            root = study / "runs" / label
            path = root / "internal_suite_v1.jsonl"
            if not path.exists():
                continue
            rows, digest = read(path)
            if not rows:
                continue
            history, history_hash = read(root / "history.jsonl")
            latest = rows[-1]["step"]
            events = {"model_first_99_from_available_history": first(history, lambda r: r["heldout"]["answer_accuracy"] >= .99)}
            events["probe_first_99"] = {
                name: first(rows, lambda r: r["features"][name]["linear_probe"]["heldout_accuracy"] >= .99)
                for name in rows[-1]["features"]}
            selected = {0, 1000, 10000, 20000, 25000, 30000, 31000, 32000, 33000, 34000, 35000, 36000,
                        37000, 40000, latest}
            if label == "reference-fraction20":
                selected.update(row["step"] for row in rows)
            result_path = root / "result.json"
            result = json.loads(result_path.read_text()) if result_path.exists() else None
            report["runs"][f"{study.name}/{label}"] = {
                "budget_updates": experiment.budget.updates, "training_complete": result is not None,
                "interrupted_before_first_checkpoint": (root / "interrupted_before_first_checkpoint").exists()
                    and not (root / "checkpoint.pt").exists(),
                "last_observed_checkpoint": latest, "observations": len(rows),
                "observation_step_list": [row["step"] for row in rows], "events": events,
                "source": str(path.relative_to(HERE.parents[1])), "source_prefix_sha256": digest,
                "history_prefix_sha256": history_hash,
                "observer_manifest": json.loads((root / "internal_suite_v1_manifest.json").read_text()),
                "snapshots": [compact(row) for row in rows if row["step"] in selected]}
    original_path = HERE.parent / "grokking_progress/runs/gptmini-seed1/history.jsonl"
    repeat_path = HERE / "runs/gptmini-seed1/history.jsonl"
    if original_path.exists() and repeat_path.exists():
        original = {row["step"]: row for row in read(original_path)[0]}
        repeat = {row["step"]: row for row in read(repeat_path)[0]}
        common = sorted(original.keys() & repeat.keys())
        differences = [step for step in common if any(original[step][split] != repeat[step][split]
                                                       for split in ("train", "heldout"))]
        report["repeat_canonical_comparison"] = {"observations": len(common), "through_step": common[-1],
                                                  "different_steps": differences}
    return report


if __name__ == "__main__":
    result = snapshot()
    (HERE / "internal_suite_results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for name, run in result["runs"].items():
        print(name, run["last_observed_checkpoint"], run["events"])
