#!/usr/bin/env python3
"""Project tasks: one entry point for the Lean tree and the Python lab.

    ./make.py <task> [arguments]

Lean
    lean                    build the tree (lake build)
    audit                   axioms of every Transformer.* declaration; `rests` must be 0
    index                   regenerate INDEX.md from src/ and the build
    forbidden               search src/ for native_decide, axioms and disabled linters

Python (the uv project in python/)
    setup                   install the locked environment
    test [pattern ...]      run the test suite, optionally only tests matching patterns
    blocks                  list the experiment language
    check <file>            load an experiment file and list its experiments
    show <file> <label>     print one experiment's description
    run <file> [labels]     train an experiment file's experiments, or the labeled ones
    report <file> [labels]  print what the runs recorded and what their records say, as JSON

Together
    verify                  lean, audit, index, forbidden and test: the checks before a commit
    clean                   remove Python bytecode caches

Only the standard library is used, so this runs before any setup.
"""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
PYTHON = ROOT / "python"


def execute(command, cwd=ROOT):
    print("+", " ".join(command), flush=True)
    environment = {key: value for key, value in os.environ.items() if key != "VIRTUAL_ENV"}
    subprocess.run(command, cwd=cwd, env=environment, check=True)


def lab(*arguments):
    """The lab command line, run from the repository root so file paths resolve here."""
    execute(["uv", "run", "--locked", "--project", str(PYTHON), "lab", *arguments])


def lean(_):
    execute(["lake", "build"])


def audit(_):
    execute(["lake", "env", "lean", "scripts/Axioms.lean"])


def index(_):
    execute([sys.executable, "scripts/index.py"])


def forbidden(_):
    pattern = r"native_decide\|^axiom \|linter\..* false"
    found = subprocess.run(["grep", "-rn", pattern, "src"], cwd=ROOT, capture_output=True, text=True)
    if found.stdout:
        sys.exit("Forbidden constructs in src/:\n" + found.stdout)
    print("src/ has no native_decide, axiom or disabled linter")


def setup(_):
    execute(["uv", "sync", "--locked"], cwd=PYTHON)


def test(arguments):
    patterns = [flag for pattern in arguments.patterns for flag in ("-k", pattern)]
    execute(["uv", "run", "--locked", "python", "-m", "unittest", "discover", "-s", "tests", *patterns],
            cwd=PYTHON)


def verify(arguments):
    for task in (lean, audit, index, forbidden):
        task(arguments)
    test(argparse.Namespace(patterns=[]))


def clean(_):
    for directory in (PYTHON, ROOT / "experiments", ROOT / "scripts"):
        for cache in directory.rglob("__pycache__"):
            if ".venv" not in cache.parts:
                shutil.rmtree(cache)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    tasks = parser.add_subparsers(required=True, metavar="task")
    for task in (lean, audit, index, forbidden, setup, verify, clean):
        tasks.add_parser(task.__name__).set_defaults(handler=task)
    command = tasks.add_parser("test")
    command.add_argument("patterns", nargs="*")
    command.set_defaults(handler=test)
    tasks.add_parser("blocks").set_defaults(handler=lambda _: lab("blocks"))
    command = tasks.add_parser("check")
    command.add_argument("file")
    command.set_defaults(handler=lambda arguments: lab("check", arguments.file))
    command = tasks.add_parser("show")
    command.add_argument("file")
    command.add_argument("label")
    command.set_defaults(handler=lambda arguments: lab("show", arguments.file, arguments.label))
    command = tasks.add_parser("run")
    command.add_argument("file")
    command.add_argument("labels", nargs="*")
    command.set_defaults(handler=lambda arguments: lab("run", arguments.file, *arguments.labels))
    command = tasks.add_parser("report")
    command.add_argument("file")
    command.add_argument("labels", nargs="*")
    command.set_defaults(handler=lambda arguments: lab("report", arguments.file, *arguments.labels))
    arguments = parser.parse_args()
    try:
        arguments.handler(arguments)
    except subprocess.CalledProcessError as error:
        sys.exit(error.returncode)


if __name__ == "__main__":
    main()
