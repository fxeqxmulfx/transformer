#!/usr/bin/env python3
"""Project tasks: one entry point for the Lean tree and the Python lab.

    ./run.py <task> [arguments]

Lean
    lean                          build the tree (lake build)
    kernel                        replay the project through Lean's kernel
    lint                          optional strict built-in linters and code-quality checks
    audit                         axioms of every Transformer.* declaration; `rests` must be 0
    index                         regenerate INDEX.md from src/ and the build
    forbidden                     search src/ for native_decide, axioms and disabled linters

Python (the uv project in python/; an experiment is a folder under experiments/)
    setup                         install the locked environment
    lock                          refresh the lab lockfile after dependency changes
    package                       build the standalone optimizer wheel and source distribution
    optimizer-parity              check optimizer recurrence examples with Lean's kernel
    test [pattern ...]            run the test suite, optionally only tests matching patterns
    references [--rebuild]         check bundled reference sources; optionally rebuild from pinned Git objects
    blocks                        list the experiment language
    check <experiment>            load an experiment and list its runs
    show <experiment> <label>     print one run's description
    run <experiment> [labels]     train an experiment's runs, or the labeled ones
    report <experiment> [labels]  print what the runs recorded and what their records say, as JSON
    profile <experiment> <label>  train one run afresh under Scalene's CPU profiler; print where its time goes

Together
    verify                        build, audit, index, forbidden, reference integrity and test
    papers [--dry-run]            fetch every cited arXiv paper missing from papers/
    clean                         remove Python bytecode caches

Only the standard library is used, so this runs before any setup.
"""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
PYTHON = ROOT / "python"
SCALENE = "scalene==2.3.0"


def execute(command, cwd=ROOT):
    print("+", " ".join(command), flush=True)
    environment = {key: value for key, value in os.environ.items() if key != "VIRTUAL_ENV"}
    # The artifact cache of lakefile.toml is this package's own; unset, Lake would
    # extend it to Mathlib and turn Mathlib's build read-only.
    environment["LAKE_ARTIFACT_CACHE"] = "false"
    subprocess.run(command, cwd=cwd, env=environment, check=True)


def lab(*arguments):
    """The lab command line, run from the repository root so experiment paths resolve here."""
    execute(["uv", "run", "--locked", "--project", str(PYTHON), "lab", *arguments])


def lean(_):
    execute(["lake", "build"])


def kernel(_):
    execute(["lake", "check"])


def lint(_):
    execute(["lake", "--wfail", "lint", "--builtin-lint", "--linters=linter.all", "--code-quality"])


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


def lock(_):
    execute(["uv", "lock", "--offline"], cwd=PYTHON)


def package(_):
    # Build a copy that has no access to the surrounding repository layout.
    with tempfile.TemporaryDirectory(prefix="lean-optim-package-") as temporary:
        scratch = Path(temporary)
        source = scratch / "source"
        shutil.copytree(ROOT / "optimizers", source, ignore=shutil.ignore_patterns("dist", "__pycache__", ".venv"))
        output = ROOT / "optimizers/dist"
        execute(["uv", "build", "--offline", "--project", str(source), "--out-dir", str(output)])
        wheel = next(output.glob("lean_optim-0.1.0-*.whl"))
        target = scratch / "installed"
        execute(["uv", "pip", "install", "--offline", "--no-deps", "--link-mode=copy", "--target", str(target), str(wheel)])
        execute(["uv", "run", "--locked", "python", "-I", str(ROOT / "scripts/check_optimizer_package.py"),
                 str(target), str(source / "tests")], cwd=PYTHON)


def optimizer_parity(_):
    execute(["lake", "build", "Transformer.ScheduleFree.SectionD_Optimistic",
             "Transformer.ScheduleFree.Section4_AdamW", "Transformer.Grokking.AdamW.ScalarRecurrence",
             "Transformer.MagnitudeDirection.SectionA_AMSGradCallbacks", "Transformer.AdamBeyond.Section5_AdamNC",
             "Transformer.AMSGrad.Section5_AdamX", "Transformer.DASH.Section2_Models",
             "Transformer.Magma.Section3_Algorithm", "Transformer.AdaFisher.Section3_Algorithm",
             "Transformer.Optimization.Basic"])
    execute(["lake", "env", "lean", "scripts/OptimizerParity.lean"])


def test(arguments):
    references(argparse.Namespace(rebuild=False))
    patterns = [flag for pattern in arguments.patterns for flag in ("-k", pattern)]
    execute(["uv", "run", "--locked", "python", "-m", "unittest", "discover", "-s", "tests", *patterns],
            cwd=PYTHON)
    execute(["uv", "run", "--locked", "python", "-m", "unittest", "discover", "-s", "../optimizers/tests", *patterns],
            cwd=PYTHON)


def references(arguments):
    execute([sys.executable, "scripts/reference_sources.py", *(["--rebuild"] if arguments.rebuild else [])])


def verify(arguments):
    for task in (lean, audit, index, forbidden, optimizer_parity):
        task(arguments)
    test(argparse.Namespace(patterns=[]))


def profile(arguments):
    """Train one run from scratch, in a temporary copy of its experiment, under Scalene's CPU profiler.

    The profile, of the lab's own lines, is kept beside the run's folder as runs/<label>.scalene.json.
    Scalene's GPU mode slows Python-heavy code, such as sampling splits, by two orders of magnitude.
    """
    experiment = (ROOT / arguments.experiment).resolve()
    output = experiment / "runs" / f"{arguments.label}.scalene.json"
    output.parent.mkdir(exist_ok=True)
    scalene = ["uv", "run", "--locked", "--project", str(PYTHON), "--with", SCALENE, "scalene"]
    with tempfile.TemporaryDirectory() as copy:
        shutil.copy(experiment / "experiment.py", copy)
        execute([*scalene, "run", "--cpu-only", "--profile-only", "lab/", "--program-path", str(PYTHON / "src"),
                 "-o", str(output), str(PYTHON / "src" / "lab" / "__main__.py"), "---", "run", copy,
                 arguments.label])
    execute([*scalene, "view", "--cli", "--reduced", str(output)])


def papers(arguments):
    execute([sys.executable, "scripts/papers.py", *(["--dry-run"] if arguments.dry_run else [])])


def clean(_):
    for directory in (PYTHON, ROOT / "optimizers", ROOT / "experiments", ROOT / "scripts"):
        for cache in directory.rglob("__pycache__"):
            if ".venv" not in cache.parts:
                shutil.rmtree(cache)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    tasks = parser.add_subparsers(required=True, metavar="task")
    for task in (lean, kernel, lint, audit, index, forbidden, setup, lock, package, verify, clean):
        tasks.add_parser(task.__name__).set_defaults(handler=task)
    tasks.add_parser("optimizer-parity").set_defaults(handler=optimizer_parity)
    command = tasks.add_parser("papers")
    command.add_argument("--dry-run", action="store_true")
    command.set_defaults(handler=papers)
    command = tasks.add_parser("test")
    command.add_argument("patterns", nargs="*")
    command.set_defaults(handler=test)
    command = tasks.add_parser("references")
    command.add_argument("--rebuild", action="store_true")
    command.set_defaults(handler=references)
    tasks.add_parser("blocks").set_defaults(handler=lambda _: lab("blocks"))
    command = tasks.add_parser("check")
    command.add_argument("experiment")
    command.set_defaults(handler=lambda arguments: lab("check", arguments.experiment))
    command = tasks.add_parser("show")
    command.add_argument("experiment")
    command.add_argument("label")
    command.set_defaults(handler=lambda arguments: lab("show", arguments.experiment, arguments.label))
    command = tasks.add_parser("run")
    command.add_argument("experiment")
    command.add_argument("labels", nargs="*")
    command.set_defaults(handler=lambda arguments: lab("run", arguments.experiment, *arguments.labels))
    command = tasks.add_parser("report")
    command.add_argument("experiment")
    command.add_argument("labels", nargs="*")
    command.set_defaults(handler=lambda arguments: lab("report", arguments.experiment, *arguments.labels))
    command = tasks.add_parser("profile")
    command.add_argument("experiment")
    command.add_argument("label")
    command.set_defaults(handler=profile)
    arguments = parser.parse_args()
    try:
        arguments.handler(arguments)
    except subprocess.CalledProcessError as error:
        sys.exit(error.returncode)


if __name__ == "__main__":
    main()
