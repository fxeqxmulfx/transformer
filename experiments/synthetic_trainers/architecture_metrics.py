"""Paired architecture outcomes under the prospective gates in STABILITY.md.

Setting: the modular division adaptation of Convexifying Transformers, Section
4. These finite phase, persistence and timing rules are follow-up choices.
Inputs must be complete run summaries obtained from stability_report.verify_archive;
this module does not replace archive, provenance or frozen-plan verification.
"""

import math
from pathlib import Path

from .confirmation_report import verify_confirmation


def verified_benchmark(directory):
    """Require every frozen AdamW/GPTMini repetition before architecture selection."""
    summary = verify_confirmation(Path(directory))
    if (not summary["repeatable_stable_benchmark"] or not summary["architecture_comparison_ready"]
            or summary["stable_grokking_runs"] != summary["planned_runs"]
            or any(not row["stable_grokking"] for row in summary["rows"])):
        raise ValueError("Architecture selection requires every frozen benchmark case to pass")
    if summary.get("optimizers") != ["adamw"]:
        raise ValueError("The user-selected architecture benchmark must retain AdamW")
    return summary


def timing_outcome(summary, *, control=False):
    """Retain observed timing even when the complete trajectory is ineligible."""
    assessment = summary["assessment"]
    event = assessment["long_confirmation"]
    complete = summary["complete_run"] and assessment["complete_canonical_history"]
    eligible = bool(complete and event and assessment["persistent_final_performance"]
                    and (not control or assessment["stable_grokking"]))
    observed = {kind: event.get(kind) if event else None
                for kind in ("training_seconds", "wall_seconds")}
    if eligible and any(value is None or not math.isfinite(value) or value <= 0
                        for value in observed.values()):
        raise ValueError("Eligible target timing needs measured positive finite costs")
    if complete and assessment["stable_grokking"]:
        label = "stable_grokking"
    elif complete and event and assessment["persistent_final_performance"]:
        label = "persistent_generalization_without_required_memorization_plateau"
    else:
        label = "does_not_meet_complete_persistent_target"
    return {"complete_budget": bool(complete), "phase_label": label,
            "memorization_plateau": assessment["plateau"],
            "persistent_final_performance": assessment["persistent_final_performance"],
            "stable_grokking": assessment["stable_grokking"],
            "tail_failures": len(assessment["tail_failures"]),
            "observed_target_training_seconds": observed["training_seconds"],
            "observed_target_wall_seconds": observed["wall_seconds"],
            "eligible_target_training_seconds": observed["training_seconds"] if eligible else None,
            "eligible_target_wall_seconds": observed["wall_seconds"] if eligible else None,
            "eligible_timing_support": int(eligible),
            "timing_scope": "first_long_joint_confirmation_with_complete_final_tail; no_between_sample_or_beyond_budget_guarantee"}


def paired_outcome(control, candidate, *, changed_fields):
    """Compare one complete pair without dropping failures or changing optimizer."""
    if any(not summary["complete_run"] or not summary["assessment"]["complete_canonical_history"]
           for summary in (control, candidate)):
        raise ValueError("Architecture comparison retains complete budgets for both models")
    if control["assessment"]["criterion"] != candidate["assessment"]["criterion"]:
        raise ValueError("Paired architecture criteria differ")
    left, right = control["config"], candidate["config"]
    differences = {key for key in left.keys() | right.keys() if left.get(key) != right.get(key)}
    if not differences or differences != set(changed_fields):
        raise ValueError("Architecture changes differ from the declared intervention")
    preserved = {"prime", "train_fraction", "seed", "data_seed", "optimizer", "steps", "batch_size",
                 "batch_policy", "eval_every", "learning_rate", "weight_decay", "warmup_steps",
                 "target", "patience", "device"}
    if differences & preserved or left["optimizer"] != "adamw":
        raise ValueError("Architecture pairs must preserve AdamW, task, data, seeds, training and scoring")
    base = timing_outcome(control, control=True)
    improved = timing_outcome(candidate)
    eligible_pair = bool(base["eligible_timing_support"] and improved["eligible_timing_support"])
    costs = {}
    for kind in ("training", "wall"):
        field = f"eligible_target_{kind}_seconds"
        costs[f"control_to_candidate_{kind}_time_ratio"] = (
            base[field] / improved[field] if eligible_pair else None)
    records = {}
    for name, summary, outcome in (("control", control, base), ("candidate", candidate, improved)):
        records[name] = {**outcome, "name": summary["name"], "parameters": summary["parameters"],
            "final_train_accuracy": summary["final"]["train"]["accuracy"],
            "final_heldout_accuracy": summary["final"]["heldout"]["accuracy"],
            "final_heldout_answer_loss": summary["final"]["heldout"]["answer_loss"],
            "final_heldout_EOS_loss": summary["final"]["heldout"]["EOS_loss"],
            "epochs_seen": summary["final"]["epochs_seen"],
            "training_seconds": summary["training_seconds"],
            "wall_seconds": summary["wall_seconds"], "diagnostic_seconds": summary["diagnostic_seconds"],
            "peak_cuda_allocated_bytes": summary["peak_cuda_allocated_bytes"],
            "peak_cuda_reserved_bytes": summary["peak_cuda_reserved_bytes"]}
    return {"control": records["control"], "candidate": records["candidate"],
            "changed_fields": sorted(changed_fields), "eligible_pair_support": int(eligible_pair), **costs,
            "final_heldout_accuracy_difference": records["candidate"]["final_heldout_accuracy"] -
                                                 records["control"]["final_heldout_accuracy"],
            "claim_scope": "one_frozen_pair; all_failures_retained; improvement_rule_and_all_pairs_required_for_a_campaign_claim"}
