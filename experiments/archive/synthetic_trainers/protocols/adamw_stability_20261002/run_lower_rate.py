"""Run the frozen lower-rate AdamW control after the complete quarter-split result.

Convexifying Transformers, Section 4: rate and persistence choices are explicit
follow-up adaptations. Keep the full budget and retain failed observations.
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
    expected = json.loads((protocol / "lower-rate-plan.json").read_text())
    output = Path(expected["output_directory"])
    if json.loads((output / "plan.json").read_text()) != expected:
        raise ValueError("Live lower-rate plan differs from the committed manifest")
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != expected["launcher_sha256"]:
        raise ValueError("Lower-rate launcher changed after freezing")
    worker = protocol / "archive_lower_rate.py"
    if hashlib.sha256(worker.read_bytes()).hexdigest() != expected["archive_worker_sha256"]:
        raise ValueError("Lower-rate archive worker changed after freezing")
    if confirmation_sources() != expected["analysis_and_driver_source_hashes"]:
        raise ValueError("Lower-rate analysis or confirmation sources changed")
    parent_path = protocol / "fraction25-plan.json"
    if hashlib.sha256(parent_path.read_bytes()).hexdigest() != expected["parent_fraction25_plan_sha256"]:
        raise ValueError("Parent quarter-split plan changed")
    parent = json.loads(parent_path.read_text())
    if len(expected["recipes"]) != 1 or expected["criterion"] != parent["criterion"]:
        raise ValueError("Lower-rate recipe count or criterion changed")
    config, primary = expected["recipes"][0]["config"], parent["recipes"][0]["config"]
    if (set(config) != set(primary)
            or {key for key in primary if primary[key] != config[key]} != {"learning_rate"}
            or config["learning_rate"] != .0003 or config["optimizer"] != "adamw"):
        raise ValueError("The intervention must change only learning rate to 0.0003")
    for key in ("source_hashes", "training_source_hashes", "analysis_and_driver_source_hashes",
                "instrumentation", "environment", "papers", "corpus"):
        if expected[key] != parent[key]:
            raise ValueError(f"Lower-rate {key} differs from the parent control")
    complete_path = Path(expected["parent_complete_comparison"])
    if (hashlib.sha256((complete_path / "artifact-hashes.json").read_bytes()).hexdigest()
            != expected["parent_comparison_artifact_manifest_sha256"]):
        raise ValueError("Parent comparison artifact manifest changed")
    complete = verify_comparison(complete_path)
    if (not complete["complete_stage"] or complete["completed_runs"] != 1
            or complete["ready_to_freeze_independent_confirmation"]
            or not complete["rows"][0]["legacy_plateau_then_generalization"]
            or not complete["rows"][0]["tail_failures"]):
        raise ValueError("The lower rate requires a complete phase-eligible but persistence-failing parent")
    if json.loads((complete_path / "plan.json").read_text()) != parent:
        raise ValueError("Complete parent comparison differs from its frozen plan")
    manifest = freeze(output, expected["recipes"], DiagnosticsConfig(**expected["instrumentation"]),
                      PersistenceConfig(**expected["criterion"]), resume=True)
    return output, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    output, manifest = checked_manifest()
    if args.check_only:
        print("Frozen lower-rate AdamW recipe, complete parent, sources and unchanged criteria match.")
    else:
        run_stage(output, manifest, progress=lambda row: print(json.dumps(row), flush=True))


if __name__ == "__main__":
    main()
