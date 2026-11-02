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
    backend-contracts             capture and check finite CPU/CUDA primitive certificates
    backend-certificates          export captured binary outputs to checked Lean proof sources
    model-trace-capture           capture a complete compiled source transformer invocation
    model-trace-certificates      check all trace premises and export ordinary Lean proofs
    model-trace-refined           check separate local allowances and export refined proofs

Python (the uv project in python/; an experiment is a folder under experiments/)
    setup                         install the locked environment
    lock [--online]               refresh the lab lockfile (offline unless requested)
    package                       build the standalone optimizer wheel and source distribution
    optimizer-parity              check optimizer recurrence examples with Lean's kernel
    optimizer-lint                check all stable Ruff rules and optimizer package formatting
    optimizer-profile             measure warmed optimizer steps, optionally under Scalene
    test [pattern ...]            run the test suite, optionally only tests matching patterns
    references [--rebuild]         check bundled reference sources; optionally rebuild from pinned Git objects
    blocks                        list the experiment language
    check <experiment>            load an experiment and list its runs
    show <experiment> <label>     print one run's description
    run <experiment> [labels]     train an experiment's runs, or the labeled ones
    report <experiment> [labels]  print what the runs recorded and what their records say, as JSON
    profile <experiment> <label>  train one run afresh under Scalene's CPU profiler; print where its time goes
    scaled-continuation <action>  freeze, run or report the fixed-scale continuation to 99%
    rms-backward <action>         freeze, run or report the RMS gradient benchmark
    context-gains <action>        freeze, run or report conditional gain recovery
    adaptive-residual <action>    matched transformer comparison to 99% validation
    basis-residual <action>       matched current GPT on the original Basis tasks
    basis-gain-probe             closed-endpoint gates, trajectory norms and width oracle
    basis-state-probe <label>     training/validation replay at both Schedule-Free states
    basis-continue <decisions>    extend or censor explicit Basis chunk endpoints
    basis-recover <decision>      resume an interrupted Basis extension unchanged
    basis-control <action>        original GPTMini under the matched Schedule-Free recipe
    basis-control-continue <file> record a validation-only continuation or censoring
    basis-control-probe <label>   copied-state training/validation and head-temperature probe
    attention-temperature <action> isolated learned score gains in current GPT
    attention-temperature-continue <file> explicit continuation or censoring

Together
    verify                        build, audit, index, forbidden, reference integrity and test
    site [--port 8765]             serve the local interactive architecture diagram
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
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parent
PYTHON = ROOT / "python"
SCALENE = "scalene==2.3.0"
RUFF = "ruff==0.17.1"


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


def backend_contracts(_):
    execute(["uv", "run", "--locked", "python", str(ROOT / "review/backend_contract_check.py")], cwd=PYTHON)


def backend_certificates(_):
    execute([sys.executable, "review/backend_contract_lean.py"])


def model_trace_capture(_):
    execute(["uv", "run", "--locked", "python", str(ROOT / "review/model_trace_capture.py")], cwd=PYTHON)


def model_trace_certificates(_):
    execute([sys.executable, "review/model_trace_arithmetic.py"])
    execute([sys.executable, "review/model_trace_lean.py"])


def model_trace_refined(_):
    execute([sys.executable, "review/model_trace_refined.py"])


def setup(_):
    execute(["uv", "sync", "--locked"], cwd=PYTHON)


def lock(arguments):
    execute(["uv", "lock", *([] if arguments.online else ["--offline"])], cwd=PYTHON)


def package(_):
    # Build a copy that has no access to the surrounding repository layout.
    with tempfile.TemporaryDirectory(prefix="lean-optim-package-") as temporary:
        scratch = Path(temporary)
        source = scratch / "source"
        shutil.copytree(ROOT / "optimizers", source,
                        ignore=shutil.ignore_patterns("dist", "profiles", ".ruff_cache", "__pycache__", ".venv"))
        output = ROOT / "optimizers/dist"
        execute(["uv", "build", "--offline", "--project", str(source), "--out-dir", str(output)])
        unpacked = scratch / "unpacked"
        with tarfile.open(output / "lean_optim-0.1.0.tar.gz") as archive:
            archive.extractall(unpacked, filter="data")
        packaged = next(unpacked.iterdir())
        for required in ("tests/test_allocations.py", "benchmarks/README.md", "benchmarks/allocation_results.json"):
            if not (packaged / required).is_file():
                sys.exit(f"Source distribution is missing {required}")
        wheel = next(output.glob("lean_optim-0.1.0-*.whl"))
        target = scratch / "installed"
        execute(["uv", "pip", "install", "--offline", "--no-deps", "--link-mode=copy", "--target", str(target), str(wheel)])
        execute(["uv", "run", "--locked", "python", "-I", str(ROOT / "scripts/check_optimizer_package.py"),
                 str(target), str(packaged / "tests")], cwd=PYTHON)


def optimizer_parity(_):
    execute(["lake", "build", "Transformer.ScheduleFree.SectionD_Optimistic",
             "Transformer.ScheduleFree.Section4_AdamW", "Transformer.ScheduleFree.AdamW",
             "Transformer.ScheduleFree.Section2_Muon",
             "Transformer.Grokking.AdamW.ScalarRecurrence",
             "Transformer.MagnitudeDirection.SectionA_AMSGradCallbacks", "Transformer.AdamBeyond.Section5_AdamNC",
             "Transformer.AMSGrad.Section5_AdamX", "Transformer.DASH.Section2_Models",
             "Transformer.Magma.Section3_Algorithm", "Transformer.AdaFisher.Section3_Algorithm",
             "Transformer.Optimization.Basic", "Transformer.ScheduleFree.AMSGrad",
             "Transformer.MagnitudeDirection.SectionA_TrainingQuadratic", "Transformer.MDW"])
    execute(["lake", "env", "lean", "scripts/OptimizerParity.lean"])
    execute(["lake", "env", "lean", "scripts/ScheduleFreeAdamWParity.lean"])
    execute(["lake", "env", "lean", "scripts/ScheduleFreeAMSGradParity.lean"])
    execute(["lake", "env", "lean", "scripts/MDWParity.lean"])
    execute(["lake", "env", "lean", "scripts/ScheduleFreeMuonParity.lean"])


def optimizer_lint(_):
    command = ["uv", "run", "--locked", "--project", str(PYTHON), "--with", RUFF, "ruff"]
    execute([*command, "check", str(ROOT / "optimizers")])
    execute([*command, "format", "--check", str(ROOT / "optimizers")])


def _optimizer_profile(arguments, source=None, revision="working-tree"):
    output = ROOT / arguments.output
    output.parent.mkdir(parents=True, exist_ok=True)
    benchmark = ROOT / "optimizers/benchmarks/allocations.py"
    command = ["uv", "run", "--locked", "--project", str(PYTHON)]
    options = ["--device", arguments.device, "--output", str(output), "--source-label", revision]
    if source is not None:
        options.extend(["--source", str(source)])
    if arguments.scalene:
        execute([*command, "--with", SCALENE, "scalene", "run", "--memory",
                 "--allocation-sampling-window", "65536",
                 "--profile-only", "lean_optim/", "--program-path", str(source or ROOT / "optimizers/src"),
                 "-o", str(output.with_suffix(".scalene.json")), str(benchmark),
                 "---", *options, "--seconds", "20"])
    else:
        execute([*command, "python", str(benchmark), *options])


def optimizer_profile(arguments):
    if arguments.revision is None:
        _optimizer_profile(arguments)
        return
    revision = subprocess.run(["git", "rev-parse", "--verify", "--end-of-options",
                               f"{arguments.revision}^{{commit}}"],
                              cwd=ROOT, check=True, text=True, capture_output=True).stdout.strip()
    with tempfile.TemporaryDirectory(prefix="lean-optim-profile-") as temporary:
        scratch = Path(temporary)
        archive = scratch / "source.tar"
        with archive.open("wb") as stream:
            subprocess.run(["git", "archive", f"{revision}:optimizers/src"], cwd=ROOT,
                           stdout=stream, check=True)
        with tarfile.open(archive) as bundle:
            bundle.extractall(scratch / "source", filter="data")
        _optimizer_profile(arguments, scratch / "source", revision)


def test(arguments):
    references(argparse.Namespace(rebuild=False))
    patterns = [flag for pattern in arguments.patterns for flag in ("-k", pattern)]
    ran_tests = False
    for directory in ("tests", "../optimizers/tests"):
        try:
            execute(["uv", "run", "--locked", "python", "-m", "unittest", "discover", "-s", directory, *patterns],
                    cwd=PYTHON)
            ran_tests = True
        except subprocess.CalledProcessError as error:
            # unittest 3.14 uses 5 for an empty selection. A package-only
            # pattern must still reach the standalone suite (and vice versa).
            if not patterns or error.returncode != 5:
                raise
    if not ran_tests:
        sys.exit(5)


def references(arguments):
    execute([sys.executable, "scripts/reference_sources.py", *(["--rebuild"] if arguments.rebuild else [])])


def verify(arguments):
    for task in (lean, audit, index, forbidden, optimizer_parity, optimizer_lint):
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


def site(arguments):
    execute([sys.executable, "sites/model_architecture/build.py"])
    print(f"Local architecture: http://127.0.0.1:{arguments.port}", flush=True)
    execute([sys.executable, "-m", "http.server", str(arguments.port),
             "--bind", "127.0.0.1", "--directory", str(ROOT / "sites/model_architecture/dist")])


def scaled_continuation(arguments):
    directory = ROOT / "experiments/residual_scaled_long"
    script = "replay.py" if arguments.action == "report" else "cohort.py"
    extra = [] if arguments.action == "report" else [arguments.action]
    execute(["uv", "run", "--locked", "python", str(directory / script), *extra], cwd=PYTHON)


def rms_backward(arguments):
    execute(["uv", "run", "--locked", "python", str(ROOT / "experiments/rms_backward/cohort.py"),
             arguments.action], cwd=PYTHON)


def context_gains(arguments):
    execute(["uv", "run", "--locked", "python", str(ROOT / "experiments/context_gains/cohort.py"),
             arguments.action], cwd=PYTHON)


def adaptive_residual(arguments):
    directory = ROOT / "experiments/adaptive_residual"
    script = "profile_loop.py" if arguments.action == "profile-run" else "cohort.py"
    extra = [] if arguments.action == "profile-run" else [arguments.action]
    execute(["uv", "run", "--locked", "python", str(directory / script), *extra], cwd=PYTHON)


def basis_residual(arguments):
    execute(["uv", "run", "--locked", "python", str(ROOT / "experiments/basis_residual/cohort.py"),
             arguments.action], cwd=PYTHON)
    if arguments.action == "report":
        execute(["uv", "run", "--locked", "python",
                 str(ROOT / "experiments/basis_residual/analysis/compact_report.py")], cwd=PYTHON)
        execute(["uv", "run", "--locked", "python",
                 str(ROOT / "experiments/basis_residual/analysis/final_audit.py"), str(ROOT)], cwd=PYTHON)


def basis_gain_probe(_):
    folder = ROOT / "experiments/basis_residual/analysis"
    for name in ("gain_transfer_probe", "gain_trajectory_norms", "gain_width_oracle"):
        execute(["uv", "run", "--locked", "python", str(folder / (name + ".py"))], cwd=PYTHON)


def basis_state_probe(arguments):
    extra = [] if arguments.snapshot is None else ["--snapshot", str(Path(arguments.snapshot).resolve())]
    execute(["uv", "run", "--locked", "python",
             str(ROOT / "experiments/basis_residual/analysis/state_probe.py"), arguments.label, *extra], cwd=PYTHON)


def basis_continue(arguments):
    execute(["uv", "run", "--locked", "python",
             str(ROOT / "experiments/basis_residual/analysis/continuation.py"),
             str(Path(arguments.decisions).resolve())], cwd=PYTHON)


def basis_recover(arguments):
    execute(["uv", "run", "--locked", "python",
             str(ROOT / "experiments/basis_residual/analysis/recovery.py"),
             str(Path(arguments.decision).resolve())], cwd=PYTHON)


def basis_control(arguments):
    execute(["uv", "run", "--locked", "python", str(ROOT / "experiments/basis_control/cohort.py"),
             arguments.action], cwd=PYTHON)
    if arguments.action == "report":
        execute(["uv", "run", "--locked", "python", str(ROOT / "scripts/audit_basis_study.py"),
                 str(ROOT / "experiments/basis_control")], cwd=PYTHON)


def basis_control_continue(arguments):
    execute(["uv", "run", "--locked", "python", str(ROOT / "scripts/basis_continuation.py"),
             str(ROOT / "experiments/basis_control"), str(Path(arguments.decisions).resolve())], cwd=PYTHON)


def basis_control_probe(arguments):
    extra = [] if arguments.snapshot is None else ["--snapshot", str(Path(arguments.snapshot).resolve())]
    execute(["uv", "run", "--locked", "python",
             str(ROOT / "experiments/basis_control/analysis/state_probe.py"), arguments.label, *extra], cwd=PYTHON)


def attention_temperature(arguments):
    execute(["uv", "run", "--locked", "python", str(ROOT / "experiments/attention_temperature/cohort.py"),
             arguments.action], cwd=PYTHON)
    if arguments.action == "report":
        execute(["uv", "run", "--locked", "python", str(ROOT / "scripts/audit_basis_study.py"),
                 str(ROOT / "experiments/attention_temperature")], cwd=PYTHON)


def attention_temperature_continue(arguments):
    execute(["uv", "run", "--locked", "python", str(ROOT / "scripts/basis_continuation.py"),
             str(ROOT / "experiments/attention_temperature"), str(Path(arguments.decisions).resolve())], cwd=PYTHON)


def clean(_):
    for directory in (PYTHON, ROOT / "optimizers", ROOT / "experiments", ROOT / "scripts"):
        for cache in directory.rglob("__pycache__"):
            if ".venv" not in cache.parts:
                shutil.rmtree(cache)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    tasks = parser.add_subparsers(required=True, metavar="task")
    for task in (lean, kernel, lint, audit, index, forbidden, setup, package, verify, clean):
        tasks.add_parser(task.__name__).set_defaults(handler=task)
    command = tasks.add_parser("lock")
    command.add_argument("--online", action="store_true", help="Fetch missing package metadata")
    command.set_defaults(handler=lock)
    tasks.add_parser("optimizer-parity").set_defaults(handler=optimizer_parity)
    tasks.add_parser("backend-contracts").set_defaults(handler=backend_contracts)
    tasks.add_parser("backend-certificates").set_defaults(handler=backend_certificates)
    tasks.add_parser("model-trace-capture").set_defaults(handler=model_trace_capture)
    tasks.add_parser("model-trace-certificates").set_defaults(handler=model_trace_certificates)
    tasks.add_parser("model-trace-refined").set_defaults(handler=model_trace_refined)
    tasks.add_parser("optimizer-lint").set_defaults(handler=optimizer_lint)
    command = tasks.add_parser("optimizer-profile")
    command.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    command.add_argument("--output", default="optimizers/profiles/allocations.json")
    command.add_argument("--scalene", action="store_true")
    command.add_argument("--revision", help="profile optimizer sources from a local Git commit")
    command.set_defaults(handler=optimizer_profile)
    command = tasks.add_parser("papers")
    command.add_argument("--dry-run", action="store_true")
    command.set_defaults(handler=papers)
    command = tasks.add_parser("site")
    command.add_argument("--port", type=int, default=8765)
    command.set_defaults(handler=site)
    command = tasks.add_parser("scaled-continuation")
    command.add_argument("action", choices=("freeze", "run", "report"))
    command.set_defaults(handler=scaled_continuation)
    command = tasks.add_parser("rms-backward")
    command.add_argument("action", choices=("freeze", "run", "report"))
    command.set_defaults(handler=rms_backward)
    command = tasks.add_parser("context-gains")
    command.add_argument("action", choices=("freeze", "run", "report"))
    command.set_defaults(handler=context_gains)
    command = tasks.add_parser("adaptive-residual")
    command.add_argument("action", choices=("freeze", "run", "profile-run", "report"))
    command.set_defaults(handler=adaptive_residual)
    command = tasks.add_parser("basis-residual")
    command.add_argument("action", choices=("freeze", "run", "report"))
    command.set_defaults(handler=basis_residual)
    tasks.add_parser("basis-gain-probe").set_defaults(handler=basis_gain_probe)
    command = tasks.add_parser("basis-state-probe")
    command.add_argument("label")
    command.add_argument("--snapshot")
    command.set_defaults(handler=basis_state_probe)
    command = tasks.add_parser("basis-continue")
    command.add_argument("decisions")
    command.set_defaults(handler=basis_continue)
    command = tasks.add_parser("basis-recover")
    command.add_argument("decision")
    command.set_defaults(handler=basis_recover)
    command = tasks.add_parser("basis-control")
    command.add_argument("action", choices=("freeze", "run", "report"))
    command.set_defaults(handler=basis_control)
    command = tasks.add_parser("basis-control-continue")
    command.add_argument("decisions")
    command.set_defaults(handler=basis_control_continue)
    command = tasks.add_parser("basis-control-probe")
    command.add_argument("label")
    command.add_argument("--snapshot")
    command.set_defaults(handler=basis_control_probe)
    command = tasks.add_parser("attention-temperature")
    command.add_argument("action", choices=("freeze", "run", "report"))
    command.set_defaults(handler=attention_temperature)
    command = tasks.add_parser("attention-temperature-continue")
    command.add_argument("decisions")
    command.set_defaults(handler=attention_temperature_continue)
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
