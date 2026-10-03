"""Portable prospective layout for six fresh paired normalizer comparisons.

The scientific prerequisite is the complete all-six native AdamW benchmark,
not an exploratory pair. CPU fixtures retain actual negative reference runs.
"""

import json
from pathlib import Path

from .confirmation_layout import plan_hash
from .paper_reproduction.provenance import ROOT
from .scheduled_layout import digest
from .stability_comparison import hashes
from .tagged_confirmation_layout import current_sources as benchmark_sources
from .tagged_confirmation_report import verify_confirmation, verified_benchmark


EXTRA_SOURCES = ("architecture_training.py", "architecture_report.py",
    "architecture_cohort_layout.py", "architecture_cohort_protocol.py",
    "architecture_cohort_report.py", "architecture_cohort_plots.py",
    "protocols/adamw_stability_20261002/freeze_architecture_cohort.py",
    "protocols/adamw_stability_20261002/run_architecture_cohort.py")
RULE = {"minimum_accuracy_gain": .01, "minimum_time_ratio": 1.05,
        "campaign_rule": "all_six_pairs_eligible_and_all_quality_gains_or_all_training_and_wall_ratios_pass"}
DEFAULT_TAGS = {"attention_normalization": "softmax", "learning_rate_schedule": "constant",
                "anneal_start": 150000, "anneal_end": 250000, "final_rate_factor": .1}
INITIALIZATION = "same_seed_and_parameter_shapes; normalizer_adapter_consumes_no_extra_RNG"
STATISTICAL_SCOPE = "six_crossed_heldout_pairs; no_IID_or_confidence_interval_claim"


def current_sources():
    result = benchmark_sources()
    for name in EXTRA_SOURCES:
        path = ROOT / "experiments/synthetic_trainers" / name
        result[str(path.relative_to(ROOT))] = digest(path)
    return dict(sorted(result.items()))


def select(benchmark, *, CPU_fixture=False):
    benchmark = Path(benchmark)
    origin = json.loads((benchmark / "plan.json").read_text())
    if origin["stage"] != "independent_confirmation":
        raise ValueError("Architecture selection requires all six scientific primary confirmations")
    outcome = verify_confirmation(benchmark) if CPU_fixture else verified_benchmark(benchmark)
    if CPU_fixture and (origin["scientific_run"] or not origin["CPU_fixture"]
                       or any(r["config"]["device"] != "cpu" for r in origin["recipes"])):
        raise ValueError("Explicit architecture fixtures require an actual CPU-only confirmation")
    if origin["planned_runs"] != 6 or outcome["completed_runs"] != 6:
        raise ValueError("The complete six-case benchmark reference is required")
    base = {**DEFAULT_TAGS, **origin["recipes"][0]["config"]}
    if CPU_fixture and "learning_rate_schedule" not in origin["recipes"][0]["config"]:
        # These inactive constant-schedule tags fit the actual short CPU budget.
        base.update(anneal_start=base["warmup_steps"], anneal_end=base["steps"])
    if base["optimizer"] != "adamw" or base["model"] != "gptmini" or base["attention_normalization"] != "softmax":
        raise ValueError("Architecture controls retain the native AdamW softmax benchmark")
    if not CPU_fixture and (base["steps"] != 300000 or not base["device"].startswith("cuda")):
        raise ValueError("Scientific architecture pairs retain the full 300,000-update GPU budget")
    return origin, base, outcome


def case_manifest(plan, recipe):
    keys = ("criterion", "instrumentation", "training_source_hashes", "environment",
            "papers", "frozen_utc", "git_commit", "scientific_run", "CPU_fixture")
    return {**{key: plan[key] for key in keys}, "recipes": [recipe], "planned_runs": 1,
        "corpus": recipe["corpus"], "source_hashes": plan["source_hashes"],
        "stage": "paired_architecture_case", "architecture_plan_sha256": plan_hash(plan),
        "protocol": "full_fixed_native_budget; no_target_stopping; retain_failures",
        "scope": "one_case_of_six_prospectively_frozen_heldout_normalizer_pairs"}


def validate_plan(plan, benchmark):
    origin, base, outcome = select(benchmark, CPU_fixture=plan["CPU_fixture"])
    if (plan["stage"] != "paired_architecture_cohort" or plan["planned_pairs"] != 6
            or plan["planned_runs"] != 12 or plan["model_seeds"] != [7, 8, 9]
            or plan["data_seeds"] != [4, 5] or plan["maximum_updates_per_run"] != 300000
            or plan["scientific_run"] != origin["scientific_run"] or plan["base_config"] != base
            or plan["benchmark_complete_outcome"] != outcome or plan["improvement_rule"] != RULE
            or plan["changed_fields"] != ["attention_normalization"]
            or type(plan["CPU_fixture"]) is not bool or type(plan["scientific_run"]) is not bool
            or plan["initialization_pairing"] != INITIALIZATION or plan["statistical_scope"] != STATISTICAL_SCOPE):
        raise ValueError("The paired scope, full budget, selected benchmark or prospective rule changed")
    if plan["scientific_run"] and (plan["CPU_fixture"] or not plan["archive_plots"]):
        raise ValueError("Scientific pairs require complete scientific evidence and archived curves")
    if plan["benchmark_hashes"] != hashes(Path(benchmark)):
        raise ValueError("The complete frozen benchmark evidence changed")
    for key in ("criterion", "instrumentation", "environment", "papers",
                "Lean_specification_hashes", "Lean_audit_validation_sha256"):
        if plan[key] != origin[key]:
            raise ValueError(f"Architecture {key} differs from the selected benchmark")
    required = {"experiments/synthetic_trainers/" + name for name in EXTRA_SOURCES}
    if (not required <= plan["source_hashes"].keys()
            or any(plan["source_hashes"].get(k) != v for k, v in origin["source_hashes"].items())
            or len(plan["training_source_hashes"]) != 19
            or any(plan["source_hashes"].get(k) != v for k, v in plan["training_source_hashes"].items())
            or not {"experiments/synthetic_trainers/" + name for name in
                    ("architecture_training.py", "sparsemax_attention.py", "scheduled_training.py", "scheduled_rates.py")}
                <= plan["training_source_hashes"].keys()):
        raise ValueError("All benchmark, model, native schedule and cohort sources must remain pinned")
    expected = [(seed, data) for data in plan["data_seeds"] for seed in plan["model_seeds"]]
    if len(plan["pairs"]) != 6 or len(plan["recipes"]) != 12:
        raise ValueError("The six complete paired cases are required")
    recipes = {row["name"]: row for row in plan["recipes"]}
    if len(recipes) != 12 or plan["execution_order"] != list(recipes):
        raise ValueError("Pair execution order is immutable and includes every case once")
    used = {row["corpus"][key] for row in origin["recipes"] for key in ("train_fingerprint", "heldout_fingerprint")}
    corpora = {}
    order = []
    for index, (pair, (seed, data)) in enumerate(zip(plan["pairs"], expected)):
        pair_name = f"seed-{seed}-data-{data}"
        names = {role: f"{pair_name}-{norm}" for role, norm in (("control", "softmax"), ("candidate", "sparsemax"))}
        if pair != {"name": pair_name, "model_seed": seed, "data_seed": data, **names}:
            raise ValueError("Architecture comparison reused or changed its held-out seeds")
        for role, norm in (("control", "softmax"), ("candidate", "sparsemax")):
            row = recipes[names[role]]
            if row["config"] != {**base, "seed": seed, "data_seed": data, "attention_normalization": norm}:
                raise ValueError("Paired cases changed more than the normalizer and new shared seeds")
            corpus = row["corpus"]
            if corpus["data_seed"] != data or corpora.setdefault(data, corpus) != corpus:
                raise ValueError("Paired controls and model seeds must share their exact data split")
            exemplar = origin["recipes"][0]["corpus"]
            if any(corpus[k] != v for k, v in exemplar.items()
                   if k not in ("data_seed", "train_fingerprint", "heldout_fingerprint")):
                raise ValueError("The selected benchmark corpus dimensions changed")
        roles = ("control", "candidate") if index % 2 == 0 else ("candidate", "control")
        order.extend(names[role] for role in roles)
    if order != plan["execution_order"] or len({c["train_fingerprint"] for c in corpora.values()}) != 2:
        raise ValueError("Alternating paired execution order or independent splits changed")
    if len({c["heldout_fingerprint"] for c in corpora.values()}) != 2 or used & {
            c[key] for c in corpora.values() for key in ("train_fingerprint", "heldout_fingerprint")}:
        raise ValueError("Architecture comparison reused a benchmark data fingerprint")
    return base
