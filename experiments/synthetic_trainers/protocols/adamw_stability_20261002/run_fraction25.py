"""Run the frozen mod-193 fraction intervention after the complete parent result.

Setting: Convexifying Transformers, Section 4. The fraction and persistence
choices are explicit follow-up adaptations, not exact manuscript settings.
"""

import argparse
import hashlib
import json
from pathlib import Path

from experiments.synthetic_trainers.confirmation_protocol import confirmation_sources
from experiments.synthetic_trainers.paper_reproduction.diagnostics import DiagnosticsConfig
from experiments.synthetic_trainers.persistence import PersistenceConfig
from experiments.synthetic_trainers.stability import freeze, run_stage
from experiments.synthetic_trainers.stability_comparison import verify_comparison


def checked_manifest():
    protocol = Path(__file__).resolve().parent
    expected = json.loads((protocol / "fraction25-plan.json").read_text())
    output = Path(expected["output_directory"])
    if json.loads((output / "plan.json").read_text()) != expected:
        raise ValueError("Live fraction plan differs from the committed manifest")
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != expected["launcher_sha256"]:
        raise ValueError("Fraction launcher changed after freezing")
    current_analysis = confirmation_sources()
    if current_analysis != expected["analysis_and_driver_source_hashes"]:
        raise ValueError("Fraction analysis or confirmation sources changed")
    parent_path = protocol / "larger-modulus-plan.json"
    if hashlib.sha256(parent_path.read_bytes()).hexdigest() != expected["parent_larger_modulus_plan_sha256"]:
        raise ValueError("Parent larger-modulus plan changed")
    parent = json.loads(parent_path.read_text())
    if len(expected["recipes"]) != 1 or expected["criterion"] != parent["criterion"]:
        raise ValueError("Fraction recipe count or criterion changed")
    config, primary = expected["recipes"][0]["config"], parent["recipes"][0]["config"]
    if ({key for key in primary if primary[key] != config[key]} != {"train_fraction"}
            or config["train_fraction"] != .25 or config["prime"] != 193 or config["optimizer"] != "adamw"):
        raise ValueError("The intervention must change only the training fraction to 25%")
    for key in ("source_hashes", "training_source_hashes", "instrumentation", "environment", "papers"):
        if expected[key] != parent[key]:
            raise ValueError(f"Fraction {key} differs from the parent controls")
    changed = {name for name, value in parent["analysis_and_driver_source_hashes"].items()
               if current_analysis[name] != value}
    if changed != {"experiments/synthetic_trainers/stability_report.py",
                   "experiments/synthetic_trainers/stability_comparison_plots.py"}:
        raise ValueError("Unexpected analysis changes since the completed parent")
    manifest = freeze(output, expected["recipes"], DiagnosticsConfig(**expected["instrumentation"]),
                      PersistenceConfig(**expected["criterion"]), resume=True)
    complete = verify_comparison(Path(expected["parent_complete_comparison"]))
    if (not complete["complete_stage"] or complete["completed_runs"] != 1
            or complete["ready_to_freeze_independent_confirmation"]):
        raise ValueError("The fraction control requires the complete negative parent calibration")
    if json.loads((Path(expected["parent_complete_comparison"]) / "plan.json").read_text()) != parent:
        raise ValueError("Complete parent comparison differs from its frozen plan")
    return output, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    output, manifest = checked_manifest()
    if args.check_only:
        print("Frozen mod-193 25% AdamW recipe, parent result, sources and unchanged criteria match.")
    else:
        run_stage(output, manifest, progress=lambda row: print(json.dumps(row), flush=True))


if __name__ == "__main__":
    main()
