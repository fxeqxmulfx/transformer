"""Resume the frozen midpoint calibration; never reconstruct the default grid."""

import argparse
import hashlib
import json
from pathlib import Path

from experiments.synthetic_trainers.paper_reproduction.diagnostics import DiagnosticsConfig
from experiments.synthetic_trainers.persistence import PersistenceConfig
from experiments.synthetic_trainers.stability import freeze, run_stage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    protocol = Path(__file__).resolve().parent
    expected = json.loads((protocol / "midpoint-plan.json").read_text())
    output = Path("experiments/runs/amsgradw_stability_20261002/calibration_midpoint_lr0002")
    if json.loads((output / "plan.json").read_text()) != expected:
        raise ValueError("Live midpoint manifest differs from the committed frozen plan")
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != expected["recipes"][0]["launcher_sha256"]:
        raise ValueError("Midpoint launcher changed after freezing")
    manifest = freeze(output, expected["recipes"],
                      DiagnosticsConfig(**expected["instrumentation"]),
                      PersistenceConfig(**expected["criterion"]), resume=True)
    if args.check_only:
        print("Frozen midpoint recipe, launcher, sources, corpus, environment and papers match.")
    else:
        run_stage(output, manifest, progress=lambda row: print(json.dumps(row), flush=True))


if __name__ == "__main__":
    main()
