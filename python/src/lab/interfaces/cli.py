"""The single entry point: `lab <command>` or `python -m lab <command>`.

Commands take an experiment and the labels of its runs, never configuration:
an experiment is a folder whose `experiment.py` writes every setting of every
run, beside the folder's `README.md`.

    lab blocks                           the language: slots, blocks, defaults
    lab check <experiment>               load an experiment; list its runs and how they differ
    lab show <experiment> <label>        the full description of one run
    lab run <experiment> [label ...]     train the runs, or the chosen ones
    lab report <experiment> [label ...]  what the runs recorded and what their records say, as JSON

A run lives in `runs/<label>/` in the experiment's folder. Running an
experiment again continues each unfinished run from its last checkpoint and
skips the finished ones; raising `budget.updates` extends a finished run.
Several runs to train train side by side, each in a `lab run` process of its
own on as many physical cores as its execution block asks for, a CUDA run
alone on its device (`infrastructure.farm`).
"""

import argparse
import json

from .. import dsl
from ..application.report import report_study
from ..application.study import run_study, sessions, survey
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
    study = load(arguments.experiment)
    for row in survey(study):
        changes = ", ".join(row["differs_from_first"]) or "-"
        print(f"{row['label']:<24} {row['fingerprint'][:12]}  {changes}")


def show(arguments):
    study = load(arguments.experiment)
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


def outcome(label, result):
    stop = result["stop"]
    ending = (f"finished {result['updates']} updates" if stop["reason"] == "budget"
              else f"stopped at update {stop['step']} ({stop['reason']})")
    print(f"{label} {ending}  {metrics(result['final'])}")
    if "best" in result:
        print(f"{label} best at update {result['best']['step']}  {metrics(result['best'])}")


def run(arguments):
    # Only training needs PyTorch, which the engine and the store load.
    from ..infrastructure.store import RunDirectories

    study = load(arguments.experiment)
    runs = RunDirectories(runs_root(arguments.experiment))
    training = [(label, experiment.execution)
                for label, experiment, _, done in sessions(study, arguments.labels, runs) if not done]
    if len(training) < 2:
        from ..infrastructure.engine import Engine

        for label, result in run_study(study, arguments.labels, runs, Engine(), progress).items():
            outcome(label, result)
        return
    from ..infrastructure.farm import train_apart

    failed = train_apart(arguments.experiment, training, lambda line: print(line, flush=True))
    for label, _ in study.select(arguments.labels):
        if label not in failed:
            outcome(label, runs.open(label).result())
    if failed:
        raise SystemExit(f"Training failed: {', '.join(failed)}")


def report(arguments):
    from ..infrastructure.store import RunDirectories

    study = load(arguments.experiment)
    reports = report_study(study, arguments.labels, RunDirectories(runs_root(arguments.experiment)))
    print(json.dumps(reports, indent=2, allow_nan=False))


def main(argv=None):
    parser = argparse.ArgumentParser(prog="lab", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(required=True, metavar="command")
    commands.add_parser("blocks", help="list the blocks of the language").set_defaults(handler=blocks)
    command = commands.add_parser("check", help="validate an experiment without training")
    command.add_argument("experiment")
    command.set_defaults(handler=check)
    command = commands.add_parser("show", help="print one run's description")
    command.add_argument("experiment")
    command.add_argument("label")
    command.set_defaults(handler=show)
    command = commands.add_parser("run", help="train an experiment's runs")
    command.add_argument("experiment")
    command.add_argument("labels", nargs="*", metavar="label")
    command.set_defaults(handler=run)
    command = commands.add_parser("report", help="print what an experiment's runs recorded, as JSON")
    command.add_argument("experiment")
    command.add_argument("labels", nargs="*", metavar="label")
    command.set_defaults(handler=report)
    arguments = parser.parse_args(argv)
    arguments.handler(arguments)
    return 0
