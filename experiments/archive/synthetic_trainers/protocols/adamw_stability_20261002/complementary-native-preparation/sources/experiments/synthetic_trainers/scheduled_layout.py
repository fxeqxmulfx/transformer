"""Portable prospective invariants for a conditional fixed-schedule calibration."""

from .attention_layout import current_sources as normalizer_sources, digest
from .paper_reproduction.provenance import ROOT, source_hashes
from .scheduled_rates import validate_schedule


EXTRA_SOURCES = (
    "scheduled_training.py", "scheduled_rates.py", "scheduled_integrity.py", "scheduled_report.py",
    "scheduled_layout.py", "scheduled_protocol.py", "scheduled_pair_report.py", "scheduled_plots.py",
    "protocols/adamw_stability_20261002/freeze_schedule_pair.py",
    "protocols/adamw_stability_20261002/run_schedule_pair.py",
    "protocols/adamw_stability_20261002/archive_schedule_pair.py",
    "protocols/adamw_stability_20261002/prepare_schedule_pair.py",
)


def current_sources():
    result = normalizer_sources()
    for name in EXTRA_SOURCES:
        path = ROOT / "experiments/synthetic_trainers" / name
        result[str(path.relative_to(ROOT))] = digest(path)
    return dict(sorted(result.items()))


def validate_pair_plan(plan):
    if (plan["stage"] != "conditional_fixed_schedule_calibration_pair" or plan["planned_runs"] != 2
            or plan["maximum_updates_per_run"] != 300000 or plan["campaign_architecture_claim_allowed"]):
        raise ValueError("Schedule calibration must retain both budgets, the user cap and its distinct scope")
    recipes = plan["recipes"]
    names = ["adamw-constant", "adamw-cosine-tail"]
    if [r["name"] for r in recipes] != names or plan["execution_order"] != names:
        raise ValueError("The fresh constant control and scheduled case must both remain frozen")
    left, right = [r["config"] for r in recipes]
    for config in (left, right):
        validate_schedule(config)
    if (set(left) != set(right) or {k for k in left if left[k] != right[k]} != {"learning_rate_schedule"}
            or left["learning_rate_schedule"] != "constant" or right["learning_rate_schedule"] != "cosine_tail"):
        raise ValueError("Only the declared schedule may differ")
    if (plan["criterion"]["target"] != left["target"]
            or not 0 < plan["criterion"]["tail_steps"] <= left["steps"]
            or plan["final_window"] != [left["steps"]-plan["criterion"]["tail_steps"],left["steps"]]
            or plan["recovery_window_steps"] != 10000 or 10000 % left["eval_every"]):
        raise ValueError("The scoring criterion, final window or recovery grid changed")
    if any(plan["corpus"].get(k) != left[k] for k in ("prime", "train_fraction", "data_seed")):
        raise ValueError("The paired corpus changed")
    required = {"experiments/synthetic_trainers/"+name for name in EXTRA_SOURCES}
    if not required <= plan["source_hashes"].keys():
        raise ValueError("All schedule training, validation, archive and driver sources must remain pinned")
    if (not plan["training_source_hashes"] or any(plan["source_hashes"].get(k) != v
            for k,v in plan["training_source_hashes"].items())):
        raise ValueError("Training fingerprints must match the frozen source list")
    for name in ("scheduled_training.py", "scheduled_rates.py"):
        if "experiments/synthetic_trainers/"+name not in plan["training_source_hashes"]:
            raise ValueError("Both fixed-rate paths must pin the trainer and actual rate interpretation")
    if plan["scientific_run"]:
        reference = plan["reference_plan"]
        outcome = plan["reference_complete_outcome"]
        if (not isinstance(reference,dict) or not isinstance(outcome,dict)
                or not reference["scientific_run"] or reference["planned_runs"] != 2
                or not outcome["complete_pair"] or not outcome["scientific_run"]
                or outcome["independent_confirmation_complete"]
                or outcome["outcome"]["control"]["stable_grokking"]):
            raise ValueError("Complete failed softmax exploration is required; passing controls enter independent confirmation")
        base = reference["recipes"][0]["config"]
        additions = {"learning_rate_schedule", "anneal_start", "anneal_end", "final_rate_factor"}
        if ({k:v for k,v in left.items() if k not in additions}
                != {k:v for k,v in base.items() if k != "attention_normalization"}
                or base["attention_normalization"] != "softmax"
                or left["steps"] != 300000 or not left["device"].startswith("cuda")
                or (left["anneal_start"],left["anneal_end"],left["final_rate_factor"]) != (150000,250000,.1)):
            raise ValueError("Scientific scheduling must retain the reference and fixed prospective interval")
        for key in ("criterion", "instrumentation", "corpus", "environment", "papers", "Lean_specification_hashes", "Lean_audit_validation_sha256"):
            if plan[key] != reference[key]:
                raise ValueError(f"Scientific calibration changed reference {key}")
        for name in source_hashes():
            if plan["training_source_hashes"].get(name) != reference["training_source_hashes"].get(name):
                raise ValueError("The original model, optimizer or core loop changed")
        if not plan["reference_hashes"]:
            raise ValueError("Every complete reference artifact must be retained")
    elif left["device"] != "cpu" or plan["reference_hashes"] is not None:
        raise ValueError("Pipeline fixtures must remain CPU-only and explicitly separate from scientific selection")
    return plan
