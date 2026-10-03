"""The single entry point: `lab <command>` or `python -m lab <command>`.

Commands take an experiment file and labels, never configuration: every
setting of a run is written in its experiment file.

    lab blocks                     the language: slots, blocks, defaults
    lab check <file>               load a file; list its experiments and how they differ
    lab show <file> <label>        the full description of one experiment
    lab run <file> [label ...]     train the experiments, or the chosen ones
    lab report <file> [label ...]  what the runs recorded and what their records say, as JSON

A run lives in `runs/<file stem>/<label>/` beside its file. Running a file
again continues each unfinished run from its last checkpoint and skips the
finished ones; raising `budget.updates` extends a finished run.
"""

import argparse
import json

from .. import dsl
from ..application.report import report_study
from ..application.study import run_study, survey
from ..domain.spec import blocks_of, composites, describe, kinds, signature
from ..infrastructure.loader import load, runs_root


def summary(cls):
    return (cls.__doc__ or "").strip().splitlines()[0]


def blocks(_):
    words = {getattr(dsl, name) for name in dsl.__all__}
    for kind in kinds():
        print(f"{kind.__name__}: {summary(kind)}")
        for cls in blocks_of(kind):
            if cls in words:
                print(f"    {signature(cls)}\n        {summary(cls)}")
    print("Composites:")
    for cls in composites():
        if cls in words:
            print(f"    {signature(cls)}\n        {summary(cls)}")


def check(arguments):
    study = load(arguments.file)
    for row in survey(study):
        changes = ", ".join(row["differs_from_first"]) or "-"
        print(f"{row['label']:<24} {row['fingerprint'][:12]}  {changes}")


def show(arguments):
    study = load(arguments.file)
    experiment = dict(study.select([arguments.label]))[arguments.label]
    print(json.dumps(describe(experiment), indent=2))


def number(value):
    return "nan" if value is None else f"{value:.4f}"


def metrics(row):
    """The loss, and any accuracy, of each split measured in `row`; other measurements are left out."""
    return "  ".join(f"{name} loss {number(values['loss'])}"
                     + (f" accuracy {values['accuracy']:.4f}" if "accuracy" in values else "")
                     for name, values in row.items() if isinstance(values, dict) and "loss" in values)


def progress(label, row):
    kind = "probe" if row.get("diagnostic_probe") else "step"
    print(f"{label} {kind} {row['step']}  {metrics(row)}  {row['wall_seconds']:.0f}s", flush=True)


def run(arguments):
    # Only training needs PyTorch, which the engine and the store load.
    from ..infrastructure.engine import Engine
    from ..infrastructure.store import RunDirectories

    study = load(arguments.file)
    results = run_study(study, arguments.labels, RunDirectories(runs_root(arguments.file)), Engine(), progress)
    for label, result in results.items():
        stop = result["stop"]
        ending = (f"finished {result['updates']} updates" if stop["reason"] == "budget"
                  else f"stopped at update {stop['step']} ({stop['reason']})")
        print(f"{label} {ending}  {metrics(result['final'])}")
        if "best" in result:
            print(f"{label} best at update {result['best']['step']}  {metrics(result['best'])}")


def report(arguments):
    from ..infrastructure.store import RunDirectories

    study = load(arguments.file)
    reports = report_study(study, arguments.labels, RunDirectories(runs_root(arguments.file)))
    print(json.dumps(reports, indent=2, allow_nan=False))


def main(argv=None):
    parser = argparse.ArgumentParser(prog="lab", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(required=True, metavar="command")
    commands.add_parser("blocks", help="list the blocks of the language").set_defaults(handler=blocks)
    command = commands.add_parser("check", help="validate an experiment file without training")
    command.add_argument("file")
    command.set_defaults(handler=check)
    command = commands.add_parser("show", help="print one experiment's description")
    command.add_argument("file")
    command.add_argument("label")
    command.set_defaults(handler=show)
    command = commands.add_parser("run", help="train an experiment file's experiments")
    command.add_argument("file")
    command.add_argument("labels", nargs="*", metavar="label")
    command.set_defaults(handler=run)
    command = commands.add_parser("report", help="print what an experiment file's runs recorded, as JSON")
    command.add_argument("file")
    command.add_argument("labels", nargs="*", metavar="label")
    command.set_defaults(handler=report)
    arguments = parser.parse_args(argv)
    arguments.handler(arguments)
    return 0
