"""Summaries of a measured history; they describe, they never steer training."""

import math

from .phases import sustained


def transition(history, target=.99, patience=2):
    """First train fit and first sustained held-out success.

    Source: the delayed-generalization lag of Convexifying Transformers
    (arXiv:2211.11052v1), Section 4, as `paper_reproduction.grokking`
    measured it: onset is the first of `patience` consecutive observations
    with held-out accuracy at least `target`.
    """
    fit = next((point["step"] for point in history if point["train"]["accuracy"] >= target), None)
    event = sustained(history, "heldout", target, patience)
    onset, confirmed = (event["onset"], event["confirmed"]) if event else (None, None)
    return {"train_fit_step": fit, "heldout_onset_step": onset, "heldout_confirmed_step": confirmed,
            "lag_steps": onset - fit if fit is not None and onset is not None else None,
            "delayed_generalization": fit is not None and onset is not None and onset > fit,
            "target": target, "patience": patience,
            "scope": "two_way_exhaustive_fixed_prime_arithmetic; no_length_transfer_or_causal_claim"}


def decade_slope(points):
    """The least-squares slope of log value against log step over the last decade of `points`.

    `points` are (step, value) pairs in step order; those of positive step,
    at least a tenth of the last, and positive value count. A value that
    settles has slope near zero, one decaying as a power of the step that
    power. None with fewer than two such points.

    Source: EXPERIMENT_PLAN.md, "Scalar MLP on mod 97 (2026-10-18)", where a
    run too slow for its budget still has a falling loss when it ends. For
    gradient descent on an f with an L-Lipschitz gradient, with a step under
    1 / L, whose orbit converges to a point w where the Łojasiewicz
    inequality holds with exponent a, f(x_k) - f(w) is at most C b^k for some
    b < 1 when a ≤ 1/2
    (`Transformer.Landscape.SaddleAvoidance.exists_le_mul_pow_of_isLojasiewiczAt`),
    and at most C (k + 1)^(-1 / (2a - 1)) when 1/2 < a < 1
    (`Transformer.Landscape.SaddleAvoidance.exists_le_mul_rpow_of_isLojasiewiczAt`).
    These bound the decay that the slope measures.
    """
    last = points[-1][0] if points else 0
    chosen = [(math.log(step), math.log(value)) for step, value in points
              if step > 0 and 10 * step >= last and value is not None and value > 0]
    if len({x for x, _ in chosen}) < 2:
        return None
    mean_x = sum(x for x, _ in chosen) / len(chosen)
    mean_y = sum(y for _, y in chosen) / len(chosen)
    return (sum((x - mean_x) * (y - mean_y) for x, y in chosen)
            / sum((x - mean_x) ** 2 for x, _ in chosen))


def step_curvature(history, rate):
    """The step times the norm of the Hessian at each observation, its maximum, and the first step where it
    reaches 1.

    `history` holds observations with a `step` and the extreme Hessian
    eigenvalues `hessian_min` and `hessian_max` of their `landscape`, of
    which the norm of the Hessian is max(|hessian_min|, hessian_max);
    `rate` is the step η of gradient descent. The maximum is None for an
    empty history, and the first step None when no value reaches 1.

    Source: EXPERIMENT_PLAN.md, "Scalar MLP on mod 97 (2026-10-18)", outcome
    H4, and the condition (S) of M8 in landscape_plan.md: the norm of the
    Hessian is at most L, with η L < 1, at every point of every segment
    [x_k, x_{k+1}] of the orbit. Under (S) step k lowers the loss by at
    least η (1 - η L) |∇f(x_k)|^2
    (`Transformer.Landscape.SaddleAvoidance.iterate_succ_le_sub_of_segment`),
    and the starts whose orbit converges to a strict saddle form a null set
    (`Transformer.Landscape.SaddleAvoidance.ae_not_tendsto_strictSaddle_of_segment`).
    Ritz values lie inside the spectrum, so a value of at least 1 at an
    observed x_k breaks (S) for every L. Values below 1 do not show that (S)
    holds: the observations sample the orbit, not every point of every
    segment.
    """
    values = [[point["step"], rate * max(abs(point["landscape"]["hessian_min"]), point["landscape"]["hessian_max"])]
              for point in history]
    return {"rate": rate, "values": values, "max": max((value for _, value in values), default=None),
            "first_step_reaching_one": next((step for step, value in values if value >= 1), None)}


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
