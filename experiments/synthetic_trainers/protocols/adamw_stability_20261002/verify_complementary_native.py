"""Check real complementary CPU preparation archives without importing PyTorch."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

from experiments.synthetic_trainers.complementary_report import hashes, verify_run


def verify(directory):
    directory = Path(directory)
    if json.loads((directory / "artifact-hashes.json").read_text())["files"] != hashes(directory):
        raise ValueError("Complementary CPU preparation checksums differ")
    plan = json.loads((directory / "plan.json").read_text())
    validation = json.loads((directory / "validation.json").read_text())
    if (plan["CPU_fixture"] is not True or plan["scientific_run"] is not False
            or validation["CPU_fixture"] is not True or validation["scientific_run"] is not False
            or validation["scientific_architecture_selected"] is not False or validation["scope"] != plan["scope"]):
        raise ValueError("CPU implementation preparation cannot certify a scientific effect")
    names = [case["name"] for case in plan["cases"]]
    if len(names) != 8 or len(set(names)) != 8 or [row["name"] for row in validation["cases"]] != names:
        raise ValueError("Complementary CPU preparation must retain all eight planned fixtures")
    total = 0
    for case, stored in zip(plan["cases"], validation["cases"]):
        output = directory / case["name"]
        config = json.loads((output / "config.json").read_text())
        if (config["training"] != case["training"] or config["task"] != case["task"]
                or config["model"] != plan["model"] or config["training"]["device"] != "cpu"
                or config["source_hashes"] != validation["source_hashes"]):
            raise ValueError("A complementary CPU case differs from its prospective plan")
        actual = verify_run(output)
        if any(stored[key] != value for key, value in actual.items()):
            raise ValueError("Complementary verification metadata differs from the complete case")
        if (not stored["final_and_selected_checkpoint_predictions_independently_reloaded"]
                or not stored["native_CPU_optimizer_checkpoint_reloaded"]):
            raise ValueError("Preparation receipt lacks the actual independent reload checks")
        total += actual["steps_completed"]
    if total != validation["total_completed_updates"]:
        raise ValueError("Complementary preparation update count differs from complete budgets")
    for filename, expected in validation["repository_source_snapshots"].items():
        if hashlib.sha256((directory / "sources" / filename).read_bytes()).hexdigest() != expected:
            raise ValueError("Complementary repository source snapshot differs")
    if hashlib.sha256((directory / "probe-snapshot.py").read_bytes()).hexdigest() != validation["probe_script_sha256"]:
        raise ValueError("Executed complementary probe snapshot differs")
    if "torch" in sys.modules:
        raise RuntimeError("Portable complementary verification unexpectedly imported PyTorch")
    return {"CPU_cases": len(names), "completed_updates": total, "torch_imported": False,
            "scientific_run": False, "prediction_verification": "recorded_independent_CPU_reloads; not_recomputed_here"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.directory)))


if __name__ == "__main__":
    main()
