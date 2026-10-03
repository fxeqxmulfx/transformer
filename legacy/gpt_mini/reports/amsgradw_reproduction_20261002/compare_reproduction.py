"""Compare newly trained AMSGradW runs with the preserved measured runs."""

from collections import defaultdict
import hashlib
import json
from pathlib import Path
from statistics import mean
import sys


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def compare(old_dir, new_dir):
    old_metadata = json.loads((old_dir / "metadata.json").read_text())
    new_metadata = json.loads((new_dir / "metadata.json").read_text())
    old_protocol, new_protocol = old_metadata["protocol"], new_metadata["protocol"]
    old = {row["id"]: row for row in read_rows(old_dir / "runs.jsonl") if row["method"] == "amsgradw"}
    rows = read_rows(new_dir / "runs.jsonl")
    new = {row["id"]: row for row in rows}
    assert len(rows) == len(new), "Duplicate new run identifiers"
    assert set(new) == set(old) and len(new) == 6, "Missing or unplanned run identifiers"
    assert old_protocol["config"] == new_protocol["model"], "Model settings differ"
    assert old_protocol["batch"] == new_protocol["batch"], "Batch sizes differ"
    assert old_protocol["stopping"] == new_protocol["stopping"], "Stopping policies differ"
    assert old_protocol["data_sha256"] == new_protocol["dataset"]["sha256"], "Datasets differ"
    assert old_protocol["data_boundaries"] == new_protocol["dataset"]["boundaries"], "Dataset splits differ"
    assert old_protocol["seeds"] == new_protocol["seeds"], "Seed plans differ"
    for job in new_protocol["jobs"]:
        assert job["rate"] == old_protocol["selected_rates"][job["attention"]][job["method"]], "Learning rates differ"

    checks = []
    selections = ("status", "stop_reason", "actual_steps", "best_step", "test_evaluations")
    hashes = ("initial_sha256", "batch_plan_sha256", "batch_sha256", "best_model_sha256")
    curve_fields = ("step", "new_best", "bad_checks", "divergent_checks", "stop_reason")
    for key in sorted(old):
        reference, current = old[key], new[key]
        assert reference["status"] == current["status"] == "ok", f"Failed training: {key}"
        comparison = {
            "id": key, "old_validation_loss": reference["validation_loss"],
            "new_validation_loss": current["validation_loss"],
            "old_test_loss": reference["test_loss"], "new_test_loss": current["test_loss"],
            "validation_loss_abs_delta": abs(current["validation_loss"] - reference["validation_loss"]),
            "test_loss_abs_delta": abs(current["test_loss"] - reference["test_loss"]),
            "old_actual_steps": reference["actual_steps"], "new_actual_steps": current["actual_steps"],
            "old_best_step": reference["best_step"], "new_best_step": current["best_step"],
            "selection_equal": all(reference[name] == current[name] for name in selections),
            "hashes_equal": {name: reference[name] == current[name] for name in hashes},
            "checkpoint_file_hash_equal": reference["checkpoint_sha256"] == current["checkpoint_sha256"],
            "curve_length_equal": len(reference["curves"]) == len(current["curves"]),
        }
        paired = list(zip(reference["curves"], current["curves"]))
        comparison["curve_decisions_equal"] = comparison["curve_length_equal"] and all(
            first[name] == second[name] for first, second in paired for name in curve_fields)
        curve_deltas = []
        for first, second in paired:
            for name in ("validation_loss", "train_loss"):
                if first[name] is None or second[name] is None:
                    assert first[name] == second[name], f"Changed finite curve values: {key}/{name}"
                else:
                    curve_deltas.append(abs(first[name] - second[name]))
        comparison["curve_max_abs_delta"] = max(curve_deltas, default=0.)
        comparison["losses_equal"] = comparison["validation_loss_abs_delta"] == comparison["test_loss_abs_delta"] == comparison["curve_max_abs_delta"] == 0.
        comparison["exact_training_reproduction"] = (
            comparison["selection_equal"] and all(comparison["hashes_equal"].values())
            and comparison["curve_decisions_equal"] and comparison["losses_equal"])
        checks.append(comparison)

    groups = defaultdict(list)
    for value in checks:
        groups[value["id"].split(":")[0]].append(value)
    means = {attention: {"old_test_loss_mean": mean(v["old_test_loss"] for v in values),
                         "new_test_loss_mean": mean(v["new_test_loss"] for v in values)}
             for attention, values in sorted(groups.items())}
    result = {"old_results": str(old_dir), "new_results": str(new_dir),
              "scope": "six fresh AMSGradW training runs; both attentions, seeds 0/1/2",
              "settings_and_data_equal": True,
              "all_six_exactly_reproduced": all(v["exact_training_reproduction"] for v in checks),
              "test_loss_means": means, "checks": checks,
              "old_raw_sha256": hashlib.sha256((old_dir / "runs.jsonl").read_bytes()).hexdigest(),
              "new_raw_sha256": hashlib.sha256((new_dir / "runs.jsonl").read_bytes()).hexdigest()}
    (new_dir / "reproduction.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    lines = ["# AMSGradW training reproduction", "",
             "Six complete fresh runs use the migrated application, CUDA Graphs, original model/data/settings,",
             "both attention modes and seeds 0, 1, 2. Historical runs remain unchanged.", "",
             "| Run | Old test CE | New test CE | Absolute delta | Best step old/new | Exact reproduction |",
             "| --- | ---: | ---: | ---: | --- | --- |"]
    for value in checks:
        lines.append(f"| {value['id']} | {value['old_test_loss']:.12f} | {value['new_test_loss']:.12f} | {value['test_loss_abs_delta']:.12g} | {value['old_best_step']}/{value['new_best_step']} | {value['exact_training_reproduction']} |")
    lines.extend(["", "Exact reproduction requires equal validation/training curves, stopping decisions,",
                  "initial weights, full/used batch plans, and the selected model SHA256. Wall-clock",
                  "timings and source/protocol fingerprints are expected to differ after migration.", ""])
    (new_dir / "REPRODUCTION.md").write_text("\n".join(lines))
    print(json.dumps({name: result[name] for name in ("all_six_exactly_reproduced", "test_loss_means")}, indent=2))
    return result["all_six_exactly_reproduced"]


if __name__ == "__main__":
    raise SystemExit(0 if compare(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()) else 1)
