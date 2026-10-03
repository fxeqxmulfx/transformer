"""Finite curve witnesses and delayed generalization; no causal attribution.

arXiv:1912.02292v1, Sections 5--7, and the HasDoubleDescent predicate in
Transformer.DoubleDescent.Section5_Curves require four ordered observations.
The grokking lag follows arXiv:2211.11052v1, Section 4, with an additional
length/position-transfer criterion for these synthetic algorithmic tasks.
"""

import math


def curve_witness(points, tolerance=0.0):
    """Find descent, ascent, descent at strictly increasing measured x values."""
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Curve tolerance must be finite and nonnegative")
    if any(not math.isfinite(x) or not math.isfinite(y) for x, y in points):
        raise ValueError("Curve points must be finite")
    if any(a[0] >= b[0] for a, b in zip(points, points[1:])):
        raise ValueError("Curve coordinates must be strictly increasing")
    if len(points) < 4:
        return None
    # A suffix minimum supplies the fourth point. Keep the lowest eligible
    # first valley; this gives a linear scan even for long epoch histories.
    suffix_min = [0] * len(points)
    best = len(points) - 1
    for i in reversed(range(len(points))):
        if points[i][1] < points[best][1]:
            best = i
        suffix_min[i] = best
    prefix_max, valley = 0, None
    for c, (_, y) in enumerate(points):
        if valley is not None and c < len(points) - 1:
            a, b = valley
            d = suffix_min[c + 1]
            if y > points[b][1] + tolerance and y > points[d][1] + tolerance:
                return {"indices": [a, b, c, d], "points": [points[i] for i in (a, b, c, d)],
                        "corrects_first_minimum": points[d][1] < points[b][1] - tolerance,
                        "tolerance": tolerance}
        if points[prefix_max][1] > y + tolerance and (valley is None or y < points[valley[1]][1]):
            valley = prefix_max, c
        if y > points[prefix_max][1]:
            prefix_max = c
    return None


def fit_value(metrics, metric):
    teacher = metrics.get("teacher_forced", metrics)
    if metric == "loss":
        return teacher["example_loss"]
    if metric == "example_error":
        return 1 - teacher["sequence_accuracy"]
    if metric == "token_error":
        return 1 - teacher["token_accuracy"]
    raise ValueError("Unknown interpolation metric")


def increase_witness(points, tolerance=0.0):
    """A sampled increase; on a sample-size axis this witnesses more-data harm."""
    # Reuse the same input validation, even for a two-point curve.
    curve_witness(points, tolerance)
    minimum = None
    for point in points:
        if minimum is not None and point[1] > minimum[1] + tolerance:
            return {"points": [minimum, point], "tolerance": tolerance}
        if minimum is None or point[1] < minimum[1]:
            minimum = point
    return None


def delayed_generalization(history, config, *, random_control=False):
    observed_fit = next((row for row in history if fit_value(row["train"], config.fit_metric) < config.fit_epsilon), None)
    clean_fit = next((row for row in history if fit_value(row["train_clean"], config.fit_metric) < config.fit_epsilon), None)

    def onset(require_transfer):
        start, streak = None, 0
        for row in history:
            probes = row.get("validation_ood_novel", row["validation_ood"])
            validation = row.get("validation_novel", row["validation"])
            qualifies = (not random_control and config.target is not None and validation is not None
                         and validation[config.target_metric] >= config.target)
            if require_transfer:
                qualifies = qualifies and bool(probes) and all(
                    probe is not None and probe[config.target_metric] >= config.target for probe in probes.values())
            if qualifies:
                streak += 1
                if streak == 1:
                    start = row
                if streak >= config.generalization_patience:
                    return start, row
            else:
                streak, start = 0, None
        return None, None

    start, confirmation = onset(True)
    id_start, id_confirmation = onset(False)
    lag = start["step"] - clean_fit["step"] if start and clean_fit else None
    id_lag = id_start["step"] - clean_fit["step"] if id_start and clean_fit else None
    # Whole-input separation is necessary to call this an algorithmic-transfer
    # candidate; causal prefix overlap alone is normal for language tasks.
    return {"observed_train_fit_step": observed_fit["step"] if observed_fit else None,
            "clean_train_fit_step": clean_fit["step"] if clean_fit else None,
            "generalization_step": start["step"] if start else None,
            "confirmed_at_step": confirmation["step"] if confirmation else None,
            "lag_steps": lag,
            "lag_epochs": start["epochs_seen"] - clean_fit["epochs_seen"] if start and clean_fit else None,
            "lag_training_seconds": start["training_seconds"] - clean_fit["training_seconds"] if start and clean_fit else None,
            "delayed_transfer_candidate": lag is not None and lag > 0,
            "id_generalization_step": id_start["step"] if id_start else None,
            "id_confirmed_at_step": id_confirmation["step"] if id_confirmation else None,
            "id_lag_steps": id_lag,
            "id_lag_epochs": id_start["epochs_seen"] - clean_fit["epochs_seen"] if id_start and clean_fit else None,
            "id_lag_training_seconds": id_start["training_seconds"] - clean_fit["training_seconds"] if id_start and clean_fit else None,
            "delayed_id_generalization_candidate": id_lag is not None and id_lag > 0,
            "criterion": {"fit_metric": config.fit_metric, "fit_epsilon": config.fit_epsilon,
                          "validation_target": config.target, "target_metric": config.target_metric,
                          "consecutive_observations": config.generalization_patience,
                          "requires_all_ood_probes": True, "requires_novel_inputs": True},
            "scope": "scheduled_validation_observations; transfer_evidence_not_proof_of_understanding"}
