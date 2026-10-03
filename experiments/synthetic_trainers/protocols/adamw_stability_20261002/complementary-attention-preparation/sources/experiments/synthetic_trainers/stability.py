"""Freeze and execute full-budget serial raw AMSGradW stability calibrations."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import subprocess

from .paper_reproduction.diagnostics import DiagnosticsConfig
from .paper_reproduction.grokking import RunConfig, train
from .paper_reproduction.modular_data import make_corpus
from .paper_reproduction.provenance import source_hashes
from .persistence import PersistenceConfig, assess
from .runtime import write_json
from .stability_protocol import calibration_recipes, environment, frozen_sources, paper_fingerprints


def validate_run(plan, recipe, manifest):
    expected = {"config": recipe["config"], "source_hashes": manifest["training_source_hashes"],
                "instrumentation": manifest["instrumentation"], "corpus": manifest["corpus"],
                "torch": manifest["environment"]["torch"], "gpu": manifest["environment"]["gpu"]}
    for key, value in expected.items():
        if plan.get(key) != value:
            raise ValueError(f"Run {key} differs from the frozen stability protocol")


def validate_complete(report, recipe, manifest):
    validate_run(report["plan"], recipe, manifest)
    assessment = assess(report, PersistenceConfig(**manifest["criterion"]))
    if not assessment["complete_canonical_history"]:
        raise ValueError("Completed report has an unfinished or noncanonical history")
    for point in report["history"]:
        for split, count in (("train", manifest["corpus"]["train_examples"]),
                             ("heldout", manifest["corpus"]["heldout_examples"])):
            if point[split]["examples"] != count:
                raise ValueError("Observation did not evaluate the exhaustive split")
    return assessment


def freeze(directory, recipes, diagnostics, criterion, *, resume=False):
    directory = Path(directory)
    path = directory / "plan.json"
    hashes = frozen_sources()
    if not recipes or len({row["name"] for row in recipes}) != len(recipes):
        raise ValueError("Distinct, nonempty recipes are required")
    base = RunConfig(**recipes[0]["config"])
    corpus = make_corpus(base.prime, base.train_fraction, base.data_seed).summary()
    for recipe in recipes:
        config = RunConfig(**recipe["config"])
        if make_corpus(config.prime, config.train_fraction, config.data_seed).summary() != corpus:
            raise ValueError("Calibration controls must share a frozen split")
        if config.device != base.device or config.target != criterion.target:
            raise ValueError("Paired device and prospective targets must match")
    fixed = {"recipes": recipes, "planned_runs": len(recipes), "criterion": asdict(criterion),
             "instrumentation": asdict(diagnostics), "source_hashes": hashes,
             "training_source_hashes": source_hashes(), "corpus": corpus,
             "environment": environment(base.device), "papers": paper_fingerprints()}
    if resume:
        if not path.exists():
            raise FileNotFoundError("No frozen stability manifest")
        manifest = json.loads(path.read_text())
        for key, value in fixed.items():
            if manifest.get(key) != value:
                raise ValueError(f"Frozen stability {key} changed")
        return manifest
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError("Stability output must be fresh")
    manifest = {**fixed, "frozen_utc": datetime.now(timezone.utc).isoformat(),
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "stage": "calibration_only; excludes_confirmation",
                "protocol": "single_mechanism_controls; full_fixed_budgets; no_target_stopping",
                "confirmation_gate": "complete_calibration_stable_grokking; new_independent_frozen_plan_required",
                "selection": "retain_all_failures; select_a_passing_recipe_before_independent_confirmation",
                "scope": "explicit_GPTMini/raw_AMSGradW_adaptation; not_exact_paper_timings_or_causality"}
    directory.mkdir(parents=True, exist_ok=True)
    write_json(path, manifest)
    return manifest


def run_stage(directory, manifest, *, progress=None):
    directory = Path(directory)
    if manifest["source_hashes"] != frozen_sources():
        raise ValueError("Stability sources changed after freezing")
    diagnostics = DiagnosticsConfig(**manifest["instrumentation"])
    completed = []
    with (directory / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another process is executing this stability stage") from error
        for recipe in manifest["recipes"]:
            if manifest["source_hashes"] != frozen_sources():
                raise ValueError("Stability sources changed before the next recipe")
            state = {"status": "running", "planned_runs": len(manifest["recipes"]),
                     "completed_runs": len(completed), "current_run": recipe["name"], "runs": completed}
            write_json(directory / "state.json", state)
            output = directory / recipe["name"]
            result_path = output / "measurements.json"
            try:
                if result_path.exists():
                    report = json.loads(result_path.read_text())
                else:
                    resuming = output.exists() and any(output.iterdir())
                    if resuming:
                        validate_run(json.loads((output / "plan.json").read_text()), recipe, manifest)
                    report = train(RunConfig(**recipe["config"]), output, resume=resuming,
                        diagnostics=diagnostics,
                        progress=(lambda row, name=recipe["name"]: progress({"run": name, **row})) if progress else None)
                assessment = validate_complete(report, recipe, manifest)
            except Exception as error:
                write_json(directory / "state.json", {**state, "status": "failed",
                           "error": f"{type(error).__name__}: {error}"})
                raise
            completed.append({"name": recipe["name"], "result": f"{recipe['name']}/measurements.json",
                              "assessment": assessment, "training_seconds": report["training_seconds"],
                              "wall_seconds": report["wall_seconds"], "final": report["final"]})
            if progress:
                progress({"event": "completed_run", **completed[-1]})
        state = {"status": "complete", "planned_runs": len(manifest["recipes"]),
                 "completed_runs": len(completed), "current_run": None, "runs": completed}
        write_json(directory / "state.json", state)
    return state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=150000)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args(argv)
    base = RunConfig(model="gptmini", optimizer="amsgradw", train_fraction=.5,
                     weight_decay=.1, device=args.device, steps=args.steps,
                     eval_every=args.eval_every, seed=args.seed, data_seed=args.data_seed)
    manifest = freeze(args.output, calibration_recipes(base), DiagnosticsConfig(args.eval_every, True),
                      PersistenceConfig(), resume=args.resume)
    if not args.plan_only:
        run_stage(args.output, manifest, progress=lambda row: print(json.dumps(row), flush=True))


if __name__ == "__main__":
    main()
