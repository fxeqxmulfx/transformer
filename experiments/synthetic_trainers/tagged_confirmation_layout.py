"""Portable six-case gates for the existing tagged native AdamW softmax recipes."""

import json
from pathlib import Path
from unittest.mock import patch

from . import confirmation_layout as original
from .attention_report import verify_pair as verify_normalizer
from .scheduled_pair_report import verify_pair as verify_schedule
from .scheduled_layout import current_sources as calibration_sources, digest
from .paper_reproduction.provenance import ROOT


EXTRA_SOURCES=("tagged_confirmation_layout.py","tagged_confirmation_protocol.py",
    "tagged_confirmation_report.py","tagged_confirmation_plots.py",
    "protocols/adamw_stability_20261002/freeze_tagged_confirmation.py",
    "protocols/adamw_stability_20261002/run_tagged_confirmation.py",
    "protocols/adamw_stability_20261002/prepare_tagged_confirmation.py")


def current_sources():
    result=calibration_sources()
    for name in EXTRA_SOURCES:
        path=ROOT/"experiments/synthetic_trainers"/name
        result[str(path.relative_to(ROOT))]=digest(path)
    return dict(sorted(result.items()))


def select(calibration,name,*,CPU_fixture=False):
    calibration=Path(calibration)
    origin=json.loads((calibration/"plan.json").read_text())
    if origin["stage"]=="exploratory_attention_normalizer_pair":
        outcome=verify_normalizer(calibration);mode="normalizer"
        if name!="adamw-softmax":
            raise ValueError("The primary benchmark retains softmax; sparsemax is an architecture candidate")
        passed=outcome["outcome"]["control"]["stable_grokking"]
    elif origin["stage"]=="conditional_fixed_schedule_calibration_pair":
        outcome=verify_schedule(calibration);mode="schedule"
        if name not in outcome["cases"]:
            raise ValueError("Selected schedule is absent from the complete paired reference")
        passed=outcome["cases"][name]["stable_grokking"]
    else:
        raise ValueError("Tagged confirmation requires a complete normalizer or schedule pair")
    recipe=next(row for row in origin["recipes"] if row["name"]==name)
    config=recipe["config"]
    if origin["instrumentation"].get("trace_gradients") is not True:
        raise ValueError("Tagged confirmation retains the complete gradient trace")
    if (config["model"]!="gptmini" or config["optimizer"]!="adamw"
            or config.get("attention_normalization","softmax")!="softmax"
            or not 0<config["steps"]<=300000):
        raise ValueError("The primary confirmation retains native AdamW, softmax GPTMini and the user cap")
    if CPU_fixture:
        if origin["scientific_run"] or config["device"]!="cpu":
            raise ValueError("Explicit negative pipeline fixtures must remain CPU-only")
    elif not origin["scientific_run"] or not passed or not config["device"].startswith("cuda"):
        raise ValueError("Scientific confirmation requires a complete passing primary calibration")
    if not CPU_fixture and config["steps"]!=300000:
        raise ValueError("Scientific confirmation retains the full 300,000-update budget")
    return origin,recipe,mode,outcome


def case_manifest(plan,recipe):
    return {**original.case_manifest(plan,recipe),"scientific_run":plan["scientific_run"],
            "training_mode":plan["training_mode"],"CPU_fixture":plan["CPU_fixture"]}


def validate_plan(plan,calibration):
    origin,recipe,mode,outcome=select(calibration,plan["selected_recipe"],CPU_fixture=plan["CPU_fixture"])
    if (plan["model_seeds"]!=[4,5,6] or plan["data_seeds"]!=[2,3] or plan["planned_runs"]!=6
            or plan["training_mode"]!=mode or plan["scientific_run"]!=origin["scientific_run"]
            or plan["maximum_updates_per_run"]!=300000 or plan["calibration_complete_outcome"]!=outcome
            or plan["Lean_specification_hashes"]!=origin["Lean_specification_hashes"]
            or plan["Lean_audit_validation_sha256"]!=origin["Lean_audit_validation_sha256"]):
        raise ValueError("The six-case scope, reference, native trainer, user cap or Lean provenance changed")
    required={"experiments/synthetic_trainers/"+name for name in EXTRA_SOURCES}
    if not required<=plan["source_hashes"].keys() or any(plan["source_hashes"].get(k)!=v
            for k,v in origin["source_hashes"].items()):
        raise ValueError("All calibration and tagged confirmation sources must remain pinned")
    with patch.object(original,"selected_recipe",lambda path,name:(origin,recipe)):
        return original.validate_plan(plan,calibration)
