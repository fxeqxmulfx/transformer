"""Portable invariants and source fingerprints for the user-directed normalizer pair."""

import hashlib
from pathlib import Path

from .paper_reproduction.provenance import ROOT, source_hashes


EXTRA_SOURCES = (
    "stability.py", "stability_protocol.py", "persistence.py", "paper_phases.py",
    "stability_integrity.py", "stability_analysis.py", "stability_report.py", "stability_plots.py",
    "confirmation_layout.py", "confirmation_protocol.py", "confirmation_report.py", "stability_confirmation.py",
    "stability_comparison.py", "stability_comparison_plots.py", "architecture_metrics.py",
    "stability_recovery.py", "stability_recovery_series.py", "attention_training.py", "sparsemax_attention.py",
    "attention_layout.py", "attention_protocol.py", "attention_report.py", "attention_plots.py",
    "protocols/adamw_stability_20261002/plot_recovery.py",
    "protocols/adamw_stability_20261002/freeze_attention_pair.py",
    "protocols/adamw_stability_20261002/run_attention_pair.py",
    "protocols/adamw_stability_20261002/archive_attention_pair.py",
    "protocols/adamw_stability_20261002/prepare_attention_pair.py",
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def current_sources():
    result = source_hashes()
    for name in EXTRA_SOURCES:
        path = Path(__file__).parent / name
        result[str(path.relative_to(ROOT))] = digest(path)
    return dict(sorted(result.items()))


def validate_pair_plan(plan):
    """Check the declared intervention without importing Torch or local resources."""
    if (plan["stage"] != "exploratory_attention_normalizer_pair" or plan["planned_runs"] != 2
            or plan["maximum_updates_per_run"] != 300000 or plan["campaign_architecture_claim_allowed"]):
        raise ValueError("Normalizer pair must retain two full exploratory cases and the user cap")
    recipes = plan["recipes"]
    if [r["name"] for r in recipes] != ["adamw-softmax", "adamw-sparsemax"]:
        raise ValueError("The paired control and candidate must both remain frozen")
    if plan["execution_order"] != ["adamw-sparsemax","adamw-softmax"]:
        raise ValueError("The user-directed candidate runs first, followed by its fresh paired control")
    left, right = [r["config"] for r in recipes]
    differences = {k for k in left.keys() | right.keys() if left.get(k) != right.get(k)}
    if differences != {"attention_normalization"} or set(left) != set(right):
        raise ValueError("Only attention normalization may differ")
    if left["attention_normalization"] != "softmax" or right["attention_normalization"] != "sparsemax":
        raise ValueError("Wrong normalizer labels")
    if (left["model"] != "gptmini" or left["optimizer"] != "adamw"
            or not 0 < left["steps"] <= 300000 or plan["criterion"]["target"] != left["target"]
            or not 0 < plan["criterion"]["tail_steps"] <= left["steps"]):
        raise ValueError("Native AdamW, scoring and total budgets must match")
    if plan["final_window"] != [left["steps"] - plan["criterion"]["tail_steps"], left["steps"]]:
        raise ValueError("The final persistence window changed")
    if (type(plan["recovery_window_steps"]) is not int or plan["recovery_window_steps"] != 10000
            or plan["recovery_window_steps"] % left["eval_every"]):
        raise ValueError("The additional recovery metric uses a fixed 10,000-update grid")
    expected = {"prime": left["prime"], "data_seed": left["data_seed"],
                "train_fraction": left["train_fraction"]}
    if any(plan["corpus"].get(k) != v for k, v in expected.items()):
        raise ValueError("The declared paired corpus changed")
    required = {"experiments/synthetic_trainers/" + name for name in EXTRA_SOURCES}
    if not required <= plan["source_hashes"].keys():
        raise ValueError("Every normalizer/training/analysis source must remain pinned")
    if (not plan["training_source_hashes"] or any(plan["source_hashes"].get(k) != v
            for k, v in plan["training_source_hashes"].items())):
        raise ValueError("Training fingerprints must match the frozen driver sources")
    for name in ("attention_training.py", "sparsemax_attention.py"):
        if "experiments/synthetic_trainers/" + name not in plan["training_source_hashes"]:
            raise ValueError("Both normalizer paths must pin the tagged factory and projection")
    if plan["scientific_run"]:
        reference = plan["reference_plan"]
        if not isinstance(reference,dict):
            raise ValueError("Scientific pair requires its frozen complete reference plan")
        if (left["steps"] != 300000 or not left["device"].startswith("cuda")
                or len(reference["recipes"]) != 1
                or {k:v for k,v in left.items() if k != "attention_normalization"} != reference["recipes"][0]["config"]):
            raise ValueError("Scientific pair must retain the complete 300,000-update reference configuration")
        for key in ("criterion", "instrumentation", "corpus", "environment", "papers"):
            if plan[key] != reference[key]:
                raise ValueError(f"Scientific pair changed reference {key}")
        for key, value in reference["training_source_hashes"].items():
            if plan["training_source_hashes"].get(key) != value:
                raise ValueError("The original model, optimizer or core training source changed")
        if not plan["reference_hashes"] or not plan["reference_complete_summary"]["complete_stage"]:
            raise ValueError("Scientific pair requires the complete verified reference")
        if plan["reference_complete_summary"]["independent_confirmation_complete"] is not False:
            raise ValueError("The exploratory reference is not the independent-confirmation gate")
    elif left["device"] != "cpu" or plan["reference_hashes"] is not None:
        raise ValueError("Pipeline fixtures must remain CPU-only and explicitly labeled")
    if plan["improvement_rule"] != {"quality_metric": "final_exhaustive_complete_RHS_accuracy",
                                    "minimum_accuracy_gain": .01, "minimum_time_ratio": 1.05,
                                    "timing_requires_both_training_and_wall": True}:
        raise ValueError("The prospective quality/time comparison rule changed")
    return plan
