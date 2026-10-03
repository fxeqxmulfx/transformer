"""Freeze scientific held-out architecture pairs after reviewed all-six success."""

import argparse
import json
from pathlib import Path
import subprocess

from experiments.synthetic_trainers.architecture_cohort_layout import current_sources, select
from experiments.synthetic_trainers.architecture_cohort_protocol import freeze
from experiments.synthetic_trainers.paper_reproduction.provenance import ROOT
from experiments.synthetic_trainers.runtime import write_json


def assert_committed(paths):
    relative = [str(Path(p).resolve().relative_to(ROOT)) for p in paths]
    tracked = subprocess.check_output(["git", "ls-files", "--error-unmatch", "--", *relative], cwd=ROOT, text=True)
    if not set(relative) <= set(tracked.splitlines()):
        raise ValueError("Scientific evidence and source files must be tracked")
    subprocess.run(["git", "diff", "--quiet", "HEAD", "--", *relative], cwd=ROOT,
                   check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--benchmark-validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    select(args.benchmark)  # Actual all-six proof precedes manifest writes or environment probing.
    receipt = json.loads(args.benchmark_validation.read_text())
    required = {"completed_scientific_runs": 6, "completed_updates_per_run": 300000,
        "offline_verification_passed": True, "PNG_figures_visually_reviewed": True,
        "PDF_figures_visually_reviewed": True, "trainer_terminal_session_consumed": True}
    if any(receipt.get(k) != v for k, v in required.items()):
        raise ValueError("All six complete primary results require reviewed, terminal evidence")
    files = [p for p in args.benchmark.rglob("*") if p.is_file()]
    assert_committed([*files, args.benchmark_validation, *[ROOT / p for p in current_sources()]])
    if args.manifest.exists():
        raise FileExistsError("A scientific architecture manifest requires a fresh path")
    plan = freeze(args.output, args.benchmark, render=True)
    write_json(args.manifest, plan)
    print(json.dumps({"manifest": str(args.manifest), "planned_pairs": 6, "planned_runs": 12,
                      "maximum_updates_per_run": 300000, "scientific_run": True}))


if __name__ == "__main__":
    main()
