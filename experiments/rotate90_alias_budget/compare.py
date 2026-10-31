"""Verify matched controls and retain independent 5000-update replay differences.

Run with Python's standard library after analyze.py budget. This reads recorded
specifications and results; it never modifies checkpoints or training histories.
"""

import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ESCAPE = ROOT.parent / "rotate90_alias_escape"
SOURCE = ROOT.parent / "rotate90_power_learning"


def read(path):
    return json.loads(path.read_text())


def main():
    result = read(ROOT / "results.json")
    short = {row["label"]: row for row in read(ESCAPE / "results.json")["runs"]}
    originals = {(row["prime"], row["seed"]): row for row in read(ESCAPE / "sources.json")}
    comparisons = []
    for row in result["runs"]:
        label = row["label"]
        description = read(ROOT / "runs" / label / "experiment.json")["description"]
        if label.startswith("long-"):
            prefix, suffix = label[len("long-"):].split("-", 1)
            control = ("restart" if prefix == "wd1" else prefix) + "-" + suffix
            old = read(ESCAPE / "runs" / control / "experiment.json")["description"]
            expected = copy.deepcopy(old)
            expected["budget"]["updates"] = 20000
            assert description == expected, (label, "unmatched duration control")
            before = row["at_5000"]["heldout"]["accuracy"]
            old_result = short[control]
            comparison = {"label": label, "matched_control": control,
                          "only_spec_difference": "budget.updates:5000->20000",
                          "within_run_5000_accuracy": before,
                          "within_run_20000_accuracy": row["heldout"]["accuracy"],
                          "independent_5000_control_accuracy": old_result["heldout"]["accuracy"],
                          "independent_5000_replay_delta": before - old_result["heldout"]["accuracy"]}
        else:
            old_result = originals[(row["prime"], row["seed"])]
            control = old_result["label"]
            old = read(SOURCE / "runs" / control / "experiment.json")["description"]
            expected = copy.deepcopy(old)
            expected["model"]["width"] = 64
            assert description == expected, (label, "unmatched width control")
            comparison = {"label": label, "matched_control": control,
                          "only_spec_difference": "model.width:16->64 (hidden size follows width)",
                          "fresh_width16_accuracy": old_result["heldout"]["accuracy"],
                          "fresh_width64_accuracy": row["heldout"]["accuracy"]}
        assert row["task"]["heldout_fingerprint"] == read(
            (ESCAPE if label.startswith("long-") else SOURCE) / "runs" / control / "result.json"
        )["final"]["task"]["heldout_fingerprint"]
        comparisons.append(comparison)
    (ROOT / "comparison.json").write_text(json.dumps({
        "scope": "Specifications and split fingerprints are checked. Independent CUDA runs may differ; duration effects use the same long run's 5000 and 20000 observations.",
        "comparisons": comparisons}, indent=2) + "\n")
    print(f"Verified {len(comparisons)} matched specifications and heldout fingerprints")


if __name__ == "__main__":
    main()
