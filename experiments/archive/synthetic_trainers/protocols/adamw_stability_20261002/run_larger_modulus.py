"""Run the separately frozen mod-193 task after the complete mod-97 pair."""

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
    expected = json.loads((protocol / "larger-modulus-plan.json").read_text())
    output = Path(expected["output_directory"])
    if json.loads((output / "plan.json").read_text()) != expected:
        raise ValueError("Live larger-modulus plan differs from the committed manifest")
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != expected["launcher_sha256"]:
        raise ValueError("Larger-modulus launcher changed after freezing")
    if confirmation_sources() != expected["analysis_and_driver_source_hashes"]:
        raise ValueError("Larger-modulus analysis or confirmation sources changed")
    parent_path = protocol / "optimizer-pair-plan.json"
    parent = json.loads(parent_path.read_text())
    if hashlib.sha256(parent_path.read_bytes()).hexdigest() != expected["parent_optimizer_pair_plan_sha256"]:
        raise ValueError("Parent optimizer-pair plan changed")
    if len(expected["recipes"]) != 1 or expected["criterion"] != parent["criterion"]:
        raise ValueError("Larger-modulus recipe count or criterion changed")
    config = expected["recipes"][0]["config"]
    primary = parent["recipes"][0]["config"]
    if ({key for key in primary if primary[key] != config[key]} != {"prime"}
            or config["prime"] != 193 or config["optimizer"] != "adamw"):
        raise ValueError("The task adaptation must change only the prime to 193")
    for key in ("source_hashes", "training_source_hashes", "instrumentation", "environment", "papers"):
        if expected[key] != parent[key]:
            raise ValueError(f"Larger-modulus {key} differs from the parent controls")
    manifest = freeze(output, expected["recipes"], DiagnosticsConfig(**expected["instrumentation"]),
                      PersistenceConfig(**expected["criterion"]), resume=True)
    return output, manifest, parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    output, manifest, parent = checked_manifest()
    if args.check_only:
        print("Frozen mod-193 AdamW recipe, unchanged criterion, sources, environment and papers match.")
    else:
        archive = Path(manifest["parent_complete_comparison"])
        verify_comparison(archive)
        if json.loads((archive / "plan.json").read_text()) != parent:
            raise ValueError("Complete parent archive differs from its frozen pair")
        run_stage(output, manifest, progress=lambda row: print(json.dumps(row), flush=True))


if __name__ == "__main__":
    main()
