"""Run every full paired native budget from a committed architecture manifest."""

import argparse
import json
from pathlib import Path
import subprocess

from experiments.synthetic_trainers.architecture_cohort_protocol import run_cohort, verify_live
from experiments.synthetic_trainers.architecture_cohort_layout import current_sources
from experiments.synthetic_trainers.paper_reproduction.provenance import ROOT
from experiments.synthetic_trainers.protocols.adamw_stability_20261002.freeze_architecture_cohort import assert_committed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    plan = json.loads(args.manifest.read_text())
    if not plan["scientific_run"] or plan["CPU_fixture"]:
        raise ValueError("The scientific launcher rejects CPU fixtures")
    assert_committed([args.manifest, *[ROOT / p for p in current_sources()]])
    committed = json.loads(subprocess.check_output(["git", "show", "HEAD:" +
        str(args.manifest.resolve().relative_to(ROOT))], cwd=ROOT, text=True))
    if committed != plan:
        raise ValueError("The committed architecture manifest differs")
    directory = Path(plan["output_directory"])
    verify_live(directory, plan)
    if args.check_only:
        print(json.dumps({"checked": True, "planned_runs": 12, "maximum_updates_per_run": 300000}))
    else:
        result = run_cohort(directory, plan, progress=lambda row: print(json.dumps(row), flush=True))
        print(json.dumps({"status": result["status"], "completed_runs": result["completed_runs"]}))


if __name__ == "__main__":
    main()
