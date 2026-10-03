"""Execute the frozen AdamW primary benchmark and its paired raw AMSGradW control."""

import argparse
import hashlib
import json
from pathlib import Path

from experiments.synthetic_trainers.confirmation_protocol import confirmation_sources
from experiments.synthetic_trainers.paper_reproduction.diagnostics import DiagnosticsConfig
from experiments.synthetic_trainers.persistence import PersistenceConfig
from experiments.synthetic_trainers.stability import freeze, run_stage


def checked_manifest():
    protocol = Path(__file__).resolve().parent
    expected = json.loads((protocol / "optimizer-pair-plan.json").read_text())
    output = Path(expected["output_directory"])
    if json.loads((output / "plan.json").read_text()) != expected:
        raise ValueError("Live optimizer-pair manifest differs from the committed frozen plan")
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != expected["launcher_sha256"]:
        raise ValueError("Optimizer-pair launcher changed after freezing")
    if confirmation_sources() != expected["analysis_and_driver_source_hashes"]:
        raise ValueError("Optimizer-pair analysis or confirmation sources changed")
    if expected["primary_optimizer"] != "adamw" or expected["criterion"] != dict(
            target=.99, heldout_ceiling=.1, plateau_steps=1000, plateau_observations=5,
            confirmation_observations=20, tail_steps=50000):
        raise ValueError("Primary optimizer or prospective criterion changed")
    recipes = expected["recipes"]
    if len(recipes) != 2 or [r["config"]["optimizer"] for r in recipes] != ["adamw", "amsgradw"]:
        raise ValueError("AdamW must precede its frozen raw optimizer control")
    primary, control = (r["config"] for r in recipes)
    if {k for k in primary if primary[k] != control[k]} != {"optimizer"}:
        raise ValueError("Paired recipes changed more than optimizer selection")
    manifest = freeze(output, recipes, DiagnosticsConfig(**expected["instrumentation"]),
                      PersistenceConfig(**expected["criterion"]), resume=True)
    return output, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    output, manifest = checked_manifest()
    if args.check_only:
        print("Frozen AdamW/raw AMSGradW pair, sources, criterion, environment and papers match.")
    else:
        run_stage(output, manifest, progress=lambda row: print(json.dumps(row), flush=True))


if __name__ == "__main__":
    main()
