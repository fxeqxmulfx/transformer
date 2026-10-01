"""Run paired modular-division reference and GPTMini confirmation experiments."""

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

from .paper_reproduction.grokking import RunConfig, train
from .paper_reproduction.modular_data import make_corpus
from .runtime import write_json


PAIRS = (("reference", "adamw"), ("gptmini", "adamw"), ("gptmini", "amsgradw"))


def recipes(base, seeds):
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Distinct initialization seeds are required")
    return [{"name": f"{model}-{optimizer}-seed{seed}",
             "config": asdict(replace(base, model=model, optimizer=optimizer, seed=seed))}
            for model, optimizer in PAIRS for seed in seeds]


def source_hashes():
    directory = Path(__file__).parent
    paths = list((directory / "paper_reproduction").glob("*.py")) + [Path("experiments/gpt_mini.py"),
        Path("experiments/optimizer_benchmark/coordinate.py"), Path("experiments/optimizer_benchmark/common.py")]
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(paths)}


def validate_run(plan, recipe, hashes):
    if plan["config"] != recipe["config"]:
        raise ValueError("Existing run configuration differs from the frozen recipe")
    if plan["source_hashes"] != hashes:
        raise ValueError("Existing run sources differ from the frozen confirmation protocol")


def run_campaign(directory, base, seeds=(0, 1, 2), *, resume=False, adopt_completed=False, progress=None):
    directory = Path(directory)
    manifest_path = directory / "plan.json"
    runs = recipes(base, seeds)
    hashes = source_hashes()
    nonempty = directory.exists() and any(directory.iterdir())
    if nonempty and not resume and not adopt_completed:
        raise FileExistsError("Use a fresh directory, resume, or explicitly adopt verified completed runs")
    if resume and not manifest_path.exists():
        raise FileNotFoundError("No frozen campaign manifest to resume")
    if resume:
        manifest = json.loads(manifest_path.read_text())
        if manifest["recipes"] != runs or manifest["source_hashes"] != hashes:
            raise ValueError("Campaign configuration or sources changed")
    else:
        names = {recipe["name"] for recipe in runs}
        if nonempty and any(path.name not in names for path in directory.iterdir()):
            raise ValueError("Adoption directory contains unrelated files")
        adopted = []
        if nonempty:
            for recipe in runs:
                path = directory / recipe["name"] / "measurements.json"
                if path.parent.exists():
                    if not path.exists():
                        raise ValueError("Only completed matching runs can be adopted")
                    report = json.loads(path.read_text())
                    validate_run(report["plan"], recipe, hashes)
                    if report["plan"]["status"] != "complete" or report["completed_steps"] != recipe["config"]["steps"]:
                        raise ValueError("Cannot adopt an unfinished run")
                    adopted.append(recipe["name"])
        manifest = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
                    "recipes": runs, "planned_runs": len(runs), "source_hashes": hashes,
                    "corpus": make_corpus(base.prime, base.train_fraction, base.data_seed).summary(),
                    "adopted_completed_runs": adopted,
                    "protocol": "paired_three_model_optimizer_controls; full_fixed_budgets; no_test_selection",
                    "scope": "reference_effect_reproduction_and_explicit_GPTMini_adaptations"}
    directory.mkdir(parents=True, exist_ok=True)
    manifest["status"] = "running"
    write_json(manifest_path, manifest)
    completed = []
    started = time.perf_counter()
    for recipe in runs:
        path = directory / recipe["name"]
        config = RunConfig(**recipe["config"])
        result_path = path / "measurements.json"
        if result_path.exists():
            report = json.loads(result_path.read_text())
            validate_run(report["plan"], recipe, hashes)
            if report["plan"]["status"] != "complete" or report["completed_steps"] != config.steps:
                raise ValueError("Completed report does not match the frozen budget")
        else:
            if path.exists() and any(path.iterdir()):
                if not resume:
                    raise ValueError("Only resume can continue an incomplete run")
                validate_run(json.loads((path / "plan.json").read_text()), recipe, hashes)
            report = train(config, path, resume=path.exists() and any(path.iterdir()),
                           progress=(lambda row, name=recipe["name"]: progress({"run": name, **row})) if progress else None)
        completed.append({"name": recipe["name"], "result": str(result_path),
                          "training_seconds": report["training_seconds"], "wall_seconds": report["wall_seconds"],
                          "transition": report["transition"], "final": report["final"]})
        write_json(directory / "campaign.json", {"planned_runs": len(runs), "completed_runs": len(completed),
                   "complete": len(completed) == len(runs), "runs": completed})
        if progress:
            progress({"event": "completed_run", "run": recipe["name"], "transition": report["transition"]})
    manifest.update(status="complete", finished_utc=datetime.now(timezone.utc).isoformat(),
                    campaign_session_seconds=time.perf_counter() - started)
    write_json(manifest_path, manifest)
    return completed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--steps", type=int, default=150000)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--adopt-completed", action="store_true")
    args = parser.parse_args(argv)
    base = RunConfig(device=args.device, steps=args.steps, eval_every=args.eval_every)
    run_campaign(args.output, base, args.seeds, resume=args.resume, adopt_completed=args.adopt_completed,
                 progress=lambda row: print(json.dumps(row), flush=True))


if __name__ == "__main__":
    main()
