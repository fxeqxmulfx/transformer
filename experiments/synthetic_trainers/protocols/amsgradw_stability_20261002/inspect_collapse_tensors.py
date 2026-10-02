"""Post hoc tensor diagnostics for Convexifying Transformers, Section 4 controls."""

import argparse
import hashlib
import json
import math
from pathlib import Path

from experiments.synthetic_trainers.stability_comparison import verify_comparison


def tensor_summary(point, diagnostic):
    tensors = diagnostic["parameters"]
    squared_gradient = math.fsum(row["gradient_l2"] ** 2 for row in tensors.values())
    assert math.isclose(math.sqrt(squared_gradient), diagnostic["gradient_l2"], rel_tol=2e-5)
    squared_weights = math.fsum(row["parameter_before_l2"] ** 2 for row in tensors.values())
    squared_updates = math.fsum(row["update_l2"] ** 2 for row in tensors.values())
    temperatures = [x for row in diagnostic["temperatures"].values()
                    for x in row["inverse_temperature"]]
    ranked = sorted(tensors, key=lambda name: tensors[name]["gradient_l2"], reverse=True)
    per_tensor = {name: {**row, "squared_gradient_share": row["gradient_l2"] ** 2 / squared_gradient,
                        "update_to_weight_ratio": row["update_l2"] / row["parameter_before_l2"]}
                  for name, row in tensors.items()}
    assert math.isclose(math.fsum(row["squared_gradient_share"] for row in per_tensor.values()), 1)
    return {"step": point["step"], "train_accuracy": point["train"]["accuracy"],
            "heldout_accuracy": point["heldout"]["accuracy"],
            "train_answer_loss": point["train"]["answer_loss"],
            "heldout_answer_loss": point["heldout"]["answer_loss"],
            "diagnostic_batch_size": diagnostic["batch_size"],
            "gradient_l2": diagnostic["gradient_l2"],
            "update_to_weight_ratio": math.sqrt(squared_updates / squared_weights),
            "inverse_temperature_range": [min(temperatures), max(temperatures)],
            "largest_gradient_tensors": ranked[:3], "parameters": per_tensor}


def inspect(archive):
    summary = verify_comparison(archive)
    assert summary["complete_stage"]
    cases = []
    for recipe in json.loads((archive / "plan.json").read_text())["recipes"]:
        source = archive / "runs" / recipe["name"]
        report = json.loads((source / "measurements.json").read_text())
        history = report["history"]
        tail = [p for p in history if p["step"] >= report["completed_steps"] - 50000]
        worst = min(tail, key=lambda p: p["heldout"]["accuracy"])
        reference = next((p for p in reversed(history)
                          if p["step"] < worst["step"] and p["train"]["accuracy"] >= .99), None)
        diagnostics = {row["step"]: row for line in (source / "diagnostics.jsonl").read_text().splitlines()
                       if line and (row := json.loads(line))}
        cases.append({"name": recipe["name"], "config": recipe["config"],
                      "source_artifact_manifest_sha256": hashlib.sha256(
                          (source / "artifact-hashes.json").read_bytes()).hexdigest(),
                      "worst_tail": tensor_summary(worst, diagnostics[worst["step"]]),
                      "earlier_train_fit": tensor_summary(reference, diagnostics[reference["step"]])
                      if reference else None})
    return {"scope": "post_hoc_selected_observations; not_independent_repeats_or_causal_evidence",
            "selection": "lowest_heldout_canonical_point_in_last_50000_updates; latest_earlier_train_accuracy_ge_99",
            "diagnostic_timing": "batch_gradients_before_update; weights_and_temperature_after_update; full_split_scores_after_update",
            "gradient_share": "squared_tensor_gradient_norm_divided_by_sum_of_squared_tensor_gradient_norms",
            "comparison_archive": str(archive),
            "comparison_artifact_manifest_sha256": hashlib.sha256(
                (archive / "artifact-hashes.json").read_bytes()).hexdigest(),
            "analysis_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "cases": cases}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Inspection destination must be fresh")
    archive = Path("experiments/synthetic_trainers/baselines/amsgradw_stability_calibration_20261002")
    result = inspect(archive)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"completed_cases": len(result["cases"]), "output": str(args.output)}))


if __name__ == "__main__":
    main()
