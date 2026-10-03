"""Summaries of a measured history; they describe, they never steer training."""

import math


def transition(history, target=.99, patience=2):
    """First train fit and first sustained held-out success.

    Source: the delayed-generalization lag of Convexifying Transformers
    (arXiv:2211.11052v1), Section 4, as `paper_reproduction.grokking`
    measured it: onset is the first of `patience` consecutive observations
    with held-out accuracy at least `target`.
    """
    fit = next((point["step"] for point in history if point["train"]["accuracy"] >= target), None)
    confirmed, onset = None, None
    for start in range(len(history) - patience + 1):
        streak = history[start:start + patience]
        if all(point["heldout"]["accuracy"] >= target for point in streak):
            onset, confirmed = streak[0]["step"], streak[-1]["step"]
            break
    return {"train_fit_step": fit, "heldout_onset_step": onset, "heldout_confirmed_step": confirmed,
            "lag_steps": onset - fit if fit is not None and onset is not None else None,
            "delayed_generalization": fit is not None and onset is not None and onset > fit,
            "target": target, "patience": patience,
            "scope": "two_way_exhaustive_fixed_prime_arithmetic; no_length_transfer_or_causal_claim"}


def milestones(history, split, percents=(50, 75, 90, 95, 99)):
    """The first observation whose accuracy on `split` reaches each percentage, and whether it held.

    A port of `summarize_epochs` of the convex MQAR comparison
    (`experiments/convex_mqar/src/convex_mqar/milestones.py`) at the
    resolution of observations, without times: a percentage is reached when
    100 correct >= percent queries; a crossing records the observation before
    it and whether every later observation reached the percentage too. A
    percentage never reached is None.
    """
    found = {}
    for percent in percents:
        reached = [100 * row[split]["correct"] >= percent * row[split]["queries"] for row in history]
        first = next((index for index, hit in enumerate(reached) if hit), None)
        previous = history[first - 1] if first else None
        found[str(percent)] = None if first is None else {
            "step": history[first]["step"], "accuracy": history[first][split]["accuracy"],
            "previous_step": None if previous is None else previous["step"],
            "previous_accuracy": None if previous is None else previous[split]["accuracy"],
            "sustained_to_end": all(reached[first:])}
    return found


def curve_witness(points, tolerance=0.0):
    """Descent, ascent, descent at strictly increasing x: a double-descent witness.

    arXiv:1912.02292v1, Sections 5--7; the HasDoubleDescent predicate in
    Transformer.DoubleDescent.Section5_Curves requires four ordered points.
    """
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Curve tolerance must be finite and nonnegative")
    if any(not math.isfinite(x) or not math.isfinite(y) for x, y in points):
        raise ValueError("Curve points must be finite")
    if any(a[0] >= b[0] for a, b in zip(points, points[1:])):
        raise ValueError("Curve coordinates must be strictly increasing")
    if len(points) < 4:
        return None
    # A suffix minimum supplies the fourth point; keeping the lowest eligible
    # first valley makes one linear scan enough.
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
