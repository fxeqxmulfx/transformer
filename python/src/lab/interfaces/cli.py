"""The single entry point: `lab <command>` or `python -m lab <command>`.

Commands take an experiment file and labels, never configuration: every
setting of a run is written in its experiment file.

    lab blocks                     the language: slots, blocks, defaults
    lab check <file>               load a file; list its experiments and how they differ
    lab show <file> <label>        the full description of one experiment
"""

import argparse
import json

from .. import dsl
from ..application.study import survey
from ..domain.spec import blocks_of, composites, describe, kinds, signature
from ..infrastructure.loader import load


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
    arguments = parser.parse_args(argv)
    arguments.handler(arguments)
    return 0
