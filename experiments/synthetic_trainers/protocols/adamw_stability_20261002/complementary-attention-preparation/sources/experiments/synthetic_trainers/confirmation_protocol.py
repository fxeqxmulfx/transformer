"""Freeze all independent modular repeats before executing the first case."""

from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

from .confirmation_layout import case_manifest, selected_recipe, validate_plan
from .paper_reproduction.grokking import RunConfig
from .paper_reproduction.modular_data import make_corpus
from .paper_reproduction.provenance import ROOT, source_hashes
from .stability_comparison import hashes
from .stability_protocol import environment, frozen_sources, paper_fingerprints
from .stability_report import analysis_hashes, write_json


def confirmation_sources():
    result = {**frozen_sources(), **analysis_hashes()}
    for name in ("confirmation_layout.py", "confirmation_protocol.py", "confirmation_report.py",
                 "stability_confirmation.py", "stability_comparison.py", "stability_comparison_plots.py"):
        path = Path(__file__).parent / name
        result[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return dict(sorted(result.items()))


def verify_live(directory, plan):
    directory = Path(directory)
    if json.loads((directory / "plan.json").read_text()) != plan:
        raise ValueError("Frozen confirmation root plan changed")
    if plan["source_hashes"] != confirmation_sources():
        raise ValueError("Frozen confirmation sources changed")
    if plan["driver_source_hashes"] != frozen_sources() or plan["training_source_hashes"] != source_hashes():
        raise ValueError("Frozen confirmation training sources changed")
    if plan["environment"] != environment(plan["recipes"][0]["config"]["device"]):
        raise ValueError("Frozen confirmation environment changed")
    if plan["papers"] != paper_fingerprints():
        raise ValueError("Frozen confirmation manuscript fingerprints changed")
    validate_plan(plan, directory / "calibration")
    for recipe in plan["recipes"]:
        path = directory / "cases" / recipe["name"] / "plan.json"
        if json.loads(path.read_text()) != case_manifest(plan, recipe):
            raise ValueError("Frozen confirmation case plan changed")
    return plan


def freeze(directory, calibration, name, *, model_seeds=(4, 5, 6), data_seeds=(2, 3), render=True):
    directory, calibration = Path(directory), Path(calibration)
    if directory.exists():
        raise FileExistsError("Confirmation output must be fresh")
    origin, selected = selected_recipe(calibration, name)
    base = RunConfig(**selected["config"])
    if origin["source_hashes"] != frozen_sources() or origin["training_source_hashes"] != source_hashes():
        raise ValueError("Calibration and confirmation must use the same training sources")
    if origin["environment"] != environment(base.device) or origin["papers"] != paper_fingerprints():
        raise ValueError("Calibration and confirmation must use the same environment and manuscripts")
    recipes = []
    for data in data_seeds:
        corpus = make_corpus(base.prime, base.train_fraction, data).summary()
        for seed in model_seeds:
            recipes.append({"name": f"seed-{seed}-data-{data}",
                "config": asdict(replace(base, seed=seed, data_seed=data)), "corpus": corpus,
                "changed_mechanism": "independent_model_and_data_seeds_only"})
    plan = {"stage": "independent_confirmation", "selected_recipe": name,
        "model_seeds": list(model_seeds), "data_seeds": list(data_seeds),
        "planned_runs": len(recipes), "recipes": recipes, "archive_plots": render,
        "source_hashes": confirmation_sources(), "driver_source_hashes": frozen_sources(),
        "training_source_hashes": source_hashes(), "environment": origin["environment"],
        "papers": origin["papers"], "criterion": origin["criterion"],
        "instrumentation": origin["instrumentation"], "calibration_hashes": hashes(calibration),
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "success_rule": "all_frozen_cases_pass_stable_grokking; no_dropped_failures",
        "protocol": "full_serial_budgets; all_cases_frozen_before_first_training_update",
        "statistical_scope": "crossed_initializations_and_data_splits; not_independent_IID_runs_or_a_confidence_interval"}
    validate_plan(plan, calibration)
    directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=directory.name + "-", dir=directory.parent) as temporary:
        output = Path(temporary) / "stage"
        output.mkdir()
        shutil.copytree(calibration, output / "calibration")
        write_json(output / "plan.json", plan)
        for recipe in recipes:
            case = output / "cases" / recipe["name"]
            case.mkdir(parents=True)
            write_json(case / "plan.json", case_manifest(plan, recipe))
        verify_live(output, plan)
        output.rename(directory)
    return plan


def resume(directory):
    directory = Path(directory)
    return verify_live(directory, json.loads((directory / "plan.json").read_text()))
