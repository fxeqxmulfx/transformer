"""Describe observed collapse neighborhoods without asserting their cause.

Setting: Convexifying Transformers, Section 4. The gradient/moment/cadence
comparisons are explicit diagnostics of the raw AMSGradW GPTMini adaptation.
"""

import math

from .paper_phases import diagnose


def diagnostic_metrics(row):
    parameters = list(row["parameters"].values())
    keys = ("parameter_l2", "parameter_before_l2", "gradient_l2", "update_l2", "m_l2", "v_l2", "maximum_l2")
    joint = {key: math.sqrt(sum(norms[key] ** 2 for norms in parameters)) for key in keys}
    temperatures = [value for values in row["temperatures"].values() for value in values["inverse_temperature"]]
    return {"step": row["step"], "batch_size": row["batch_size"], "epoch_tail": row["epoch_tail"],
            "epoch_wraps_in_batch": row["epoch_wraps_in_batch"], "learning_rate": row["learning_rate"],
            "answer_loss": row["answer_loss"], "EOS_loss": row["EOS_loss"], **joint,
            "joint_relative_update": joint["update_l2"] / joint["parameter_before_l2"] if joint["parameter_before_l2"] else None,
            "maximum_buffer_min": min(norms["maximum_min"] for norms in parameters) if parameters else None,
            "maximum_buffer_max": max(norms["maximum_max"] for norms in parameters) if parameters else None,
            "temperature_min": min(temperatures) if temperatures else None,
            "temperature_max": max(temperatures) if temperatures else None}


def collapse_neighborhoods(report, probes, diagnostics):
    config, history = report["plan"]["config"], report["history"]
    event = diagnose(report)["sustained_heldout_target"]
    if event is None:
        return {"confirmed_step": None, "failures_after_confirmation": 0, "neighborhoods": [],
                "missing_neighbor_support": 0, "scope": "no_confirmed_heldout_event"}
    probe_lookup = {point["step"]: point for point in probes}
    diagnostic_lookup = {point["step"]: diagnostic_metrics(point) for point in diagnostics}
    failures = [p for p in history if p["step"] >= event["confirmed"]
                and p["heldout"]["accuracy"] < config["target"]]
    neighborhoods = []
    for point in failures:
        step = point["step"]
        before, after = probe_lookup.get(step - 1), probe_lookup.get(step + 1)
        if before is None or after is None:
            continue
        scores = {label: observation["heldout"] for label, observation in
                  (("before", before), ("canonical", point), ("after", after))}
        neighborhoods.append({"step": step, "heldout": scores,
            "batch_sizes": {"before": before["last_batch_size"], "canonical": point["last_batch_size"],
                            "after": after["last_batch_size"]},
            "isolated_at_canonical_observation": min(scores["before"]["accuracy"], scores["after"]["accuracy"]) >= config["target"],
            "still_below_target_both_neighbors": max(scores["before"]["accuracy"], scores["after"]["accuracy"]) < config["target"],
            "recovered_by_next_update": scores["after"]["accuracy"] >= config["target"],
            "diagnostics": {label: diagnostic_lookup.get(index) for label, index in
                            (("before", step - 1), ("canonical", step), ("after", step + 1))}})
    return {"confirmed_step": event["confirmed"], "failures_after_confirmation": len(failures),
            "numeric_answer_failures": sum(p["heldout"]["answer_accuracy"] < config["target"] for p in failures),
            "EOS_failures": sum(p["heldout"]["EOS_accuracy"] < config["target"] for p in failures),
            "isolated_at_canonical_observation": sum(p["isolated_at_canonical_observation"] for p in neighborhoods),
            "still_below_target_both_neighbors": sum(p["still_below_target_both_neighbors"] for p in neighborhoods),
            "recovered_by_next_update": sum(p["recovered_by_next_update"] for p in neighborhoods),
            "missing_neighbor_support": len(failures) - len(neighborhoods), "neighborhoods": neighborhoods,
            "scope": "observed_immediate_neighbors_of_canonical_failures_after_legacy_confirmation; categories_may_overlap; no_causal_claim"}


def summarize(report, assessment, probes, diagnostics, name):
    config = report["plan"]["config"]
    metrics = [diagnostic_metrics(row) for row in diagnostics]
    collapse = collapse_neighborhoods(report, probes, diagnostics)
    confirmed = collapse["confirmed_step"]
    late = [row for row in metrics if confirmed is not None and row["step"] >= confirmed]
    return {"name": name, "complete_run": True, "config": config,
            "parameters": report["plan"]["parameters"], "canonical_observations": len(report["history"]),
            "diagnostic_observations": len(diagnostics), "neighbor_observations": len(probes),
            "training_seconds": report["training_seconds"], "diagnostic_seconds": report["diagnostic_seconds"],
            "wall_seconds": report["wall_seconds"], "peak_cuda_allocated_bytes": report["peak_cuda_allocated_bytes"],
            "peak_cuda_reserved_bytes": report["peak_cuda_reserved_bytes"], "assessment": assessment,
            "final": report["final"], "collapse_diagnostics": collapse,
            "largest_post_confirmation_gradients": sorted(late, key=lambda p: p["gradient_l2"], reverse=True)[:10],
            "diagnostic_sampling": "frozen_sampled_updates; gradients_and_minibatch_losses_pre_update; weights_moments_updates_and_temperatures_post_update",
            "scope": "one_completed_exploratory_calibration; not_independent_confirmation_or_stage_completion"}
