"""What surrounded each held-out failure of a modular run after it generalized, read from its probes and diagnostics.

A port of `stability_analysis` of `experiments/synthetic_trainers`:
`collapse_neighborhoods` (`collapse`), `diagnostic_metrics`, and the largest
post-confirmation gradients of its `summarize` (`largest_gradients`). The
setting is the delayed generalization of modular division in Convexifying
Transformers (arXiv:2211.11052v1), Section 4; the neighborhoods, norms,
moments and temperatures are diagnostics of the follow-up study, not the
paper's. They describe what was observed around a failure and claim no cause.

A failure is a canonical observation, at or after the confirmation of the
first sustained held-out success, whose held-out accuracy is below target.
Its neighborhood is the probes one update before and after it, and the
diagnostics sampled at those three updates.
"""

import math

from .phases import sustained

NORMS = ("parameter_l2", "parameter_before_l2", "gradient_l2", "update_l2")
CATEGORIES = ("isolated_at_canonical_observation", "still_below_target_both_neighbors", "recovered_by_next_update")


def diagnostic_metrics(row):
    """A diagnostics row of a modular run in joint norms, each the root of the sum of squared per-tensor norms.

    The moments are the ones the row recorded: AdamW's `exp_avg` and
    `exp_avg_sq`, AMSGradW's raw `m`, `v` and `maximum`, whose buffer
    extremes are read too. The gradient norm is the joint norm of the
    gradients the update applied, not the row's own pre-clip norm.
    """
    parameters = list(row["parameters"].values())
    first = parameters[0] if parameters else {}
    keys = NORMS + tuple(key for key in first if key.endswith("_l2") and key not in NORMS)
    joint = {key: math.sqrt(sum(norms[key] ** 2 for norms in parameters)) for key in keys}
    extremes = {"maximum_buffer_min": min(norms["maximum_min"] for norms in parameters),
                "maximum_buffer_max": max(norms["maximum_max"] for norms in parameters)} if "maximum_min" in first else {}
    temperatures = [value for values in row["temperatures"].values() for value in values["inverse_temperature"]]
    return {"step": row["step"], "batch_size": row["batch_size"], "epoch_tail": row["epoch_tail"],
            "epoch_wraps_in_batch": row["epoch_wraps_in_batch"], "learning_rate": row["learning_rate"],
            "answer_loss": row["answer_loss"], "EOS_loss": row["EOS_loss"], **joint,
            "joint_relative_update": joint["update_l2"] / joint["parameter_before_l2"]
            if joint["parameter_before_l2"] else None, **extremes,
            "temperature_min": min(temperatures) if temperatures else None,
            "temperature_max": max(temperatures) if temperatures else None}


def collapse(history, probes, diagnostics, target=.99, patience=2):
    """Each held-out failure after the first sustained success, between the probes one update before and after it.

    A failure is isolated when both neighbors reach `target`, still below
    when neither does, and recovered when the next update reaches it; the
    categories may overlap. A failure lacking either neighbor probe has no
    neighborhood and counts as missing support.
    """
    event = sustained(history, "heldout", target, patience)
    if event is None:
        return {"confirmed_step": None, "failures_after_confirmation": 0, "neighborhoods": [],
                "missing_neighbor_support": 0, "scope": "no_confirmed_heldout_event"}
    probed = {point["step"]: point for point in probes}
    sampled = {row["step"]: row for row in diagnostics}
    failures = [point for point in history
                if point["step"] >= event["confirmed"] and point["heldout"]["accuracy"] < target]
    neighborhoods = []
    for point in failures:
        step = point["step"]
        observations = {"before": probed.get(step - 1), "canonical": point, "after": probed.get(step + 1)}
        if None in observations.values():
            continue
        scores = {label: observation["heldout"] for label, observation in observations.items()}
        neighbors = (scores["before"]["accuracy"], scores["after"]["accuracy"])
        neighborhoods.append({
            "step": step, "heldout": scores,
            "batch_sizes": {label: observation["last_batch_size"] for label, observation in observations.items()},
            "isolated_at_canonical_observation": min(neighbors) >= target,
            "still_below_target_both_neighbors": max(neighbors) < target,
            "recovered_by_next_update": scores["after"]["accuracy"] >= target,
            "diagnostics": {label: diagnostic_metrics(sampled[update]) if update in sampled else None
                            for label, update in (("before", step - 1), ("canonical", step), ("after", step + 1))}})
    return {"confirmed_step": event["confirmed"], "failures_after_confirmation": len(failures),
            "numeric_answer_failures": sum(point["heldout"]["answer_accuracy"] < target for point in failures),
            "EOS_failures": sum(point["heldout"]["EOS_accuracy"] < target for point in failures),
            **{category: sum(neighborhood[category] for neighborhood in neighborhoods) for category in CATEGORIES},
            "missing_neighbor_support": len(failures) - len(neighborhoods), "neighborhoods": neighborhoods,
            "scope": "observed_immediate_neighbors_of_canonical_failures_after_legacy_confirmation; "
                     "categories_may_overlap; no_causal_claim"}


def largest_gradients(diagnostics, since, count=10):
    """The `count` sampled updates from `since` on with the largest joint gradient norm, largest first.

    A run that never generalized has no `since`, and none.
    """
    late = [diagnostic_metrics(row) for row in diagnostics if since is not None and row["step"] >= since]
    return sorted(late, key=lambda metrics: metrics["gradient_l2"], reverse=True)[:count]
