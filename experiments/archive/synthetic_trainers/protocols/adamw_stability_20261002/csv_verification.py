"""Execute the frozen CSV comparison with linear rather than quadratic work.

This post-freeze implementation repair leaves training, scoring, expected
CSV cells and frozen source files unchanged. Every original archive check
still runs. The runtime adapter and its SHA256 are recorded separately.
"""

import argparse
from contextlib import contextmanager
import csv
import json
from pathlib import Path
import runpy
import sys

from experiments.synthetic_trainers import stability_report


def verify_csv(path, rows):
    """Compare the same complete tables, computing their common columns once."""
    with path.open() as stream:
        actual = list(csv.DictReader(stream))
    columns = sorted({key for row in rows for key in row})
    expected = [{key: "" if row.get(key) is None else str(row[key]) for key in columns}
                for row in rows]
    if actual != expected:
        raise ValueError(f"Archive {path.name} differs from measured rows")


@contextmanager
def linear_csv_verification():
    """Temporarily replace only the CSV table comparison in this process."""
    original = stability_report.verify_csv
    stability_report.verify_csv = verify_csv
    try:
        yield
    finally:
        stability_report.verify_csv = original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("archive", "comparison", "module"))
    parser.add_argument("target")
    parser.add_argument("module_arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.kind != "module" and args.module_arguments:
        parser.error("Only module execution accepts additional arguments")
    with linear_csv_verification():
        if args.kind == "archive":
            summary = stability_report.verify_archive(Path(args.target))
            print(json.dumps({"verified": True, "name": summary["name"],
                              "torch_imported": "torch" in sys.modules}))
        elif args.kind == "comparison":
            from experiments.synthetic_trainers.stability_comparison import verify_comparison
            summary = verify_comparison(Path(args.target))
            print(json.dumps({"verified": True, "completed_runs": summary["completed_runs"],
                              "torch_imported": "torch" in sys.modules}))
        else:
            allowed = "experiments.synthetic_trainers.protocols.adamw_stability_20261002.run_larger_modulus"
            if args.target != allowed:
                parser.error("This frozen-run adapter only executes the mod-193 launcher")
            sys.argv = [args.target, *args.module_arguments]
            runpy.run_module(args.target, run_name="__main__")


if __name__ == "__main__":
    main()
