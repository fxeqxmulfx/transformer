"""The phases of a grokking run, read from its observations.

A port of `paper_phases` of `experiments/synthetic_trainers`: the sustained
train fit, the sustained held-out success, the memorization plateau between
them, and the shape of the held-out error before generalization. It
describes a measured history and claims no cause. The setting is the delayed
generalization of Convexifying Transformers (arXiv:2211.11052v1), Section 4,
where modular division fits its training split long before its held-out
split; the near-chance ceiling is an explicit diagnostic, not a parameter of
that experiment. The error curve adapts the epoch-wise double descent of
Deep Double Descent (arXiv:1912.02292v1), Section 6, to these curves.
"""


def sustained(history, split, target, patience):
    """The first `patience` consecutive observations of `split` at accuracy `target` or more, if any."""
    for start in range(len(history) - patience + 1):
        window = history[start:start + patience]
        if all(point[split]["accuracy"] >= target for point in window):
            return {"onset": window[0]["step"], "confirmed": window[-1]["step"]}
    return None


def times(history, event):
    """The cumulative training and wall seconds at an event's observations, when each recorded both."""
    if event is None:
        return None
    by_step = {point["step"]: point for point in history}
    keys = ("training_seconds", "wall_seconds")
    if any(key not in by_step[step] for step in event.values() for key in keys):
        return None
    return {name: {key: by_step[step][key] for key in keys} for name, step in event.items()}


def error_curve(history, train, heldout, target, margin=.02):
    """Held-out error falling before the train fit, rising while train stays fitted, and falling again.

    The first minimum is taken before the sustained train fit and the peak
    while train is fitted, before the held-out onset; the curve is a full
    double descent when the initial, minimum, peak and final observations
    are in order and each change exceeds `margin`.
    """
    if train is None:
        return None
    early = [point for point in history if point["step"] < train["onset"]]
    stop = heldout["onset"] if heldout else history[-1]["step"] + 1
    fitted = [point for point in history if train["onset"] <= point["step"] < stop
              and point["train"]["accuracy"] >= target]
    if not early or not fitted:
        return None
    initial, minimum, peak, final = (history[0], max(early, key=lambda point: point["heldout"]["accuracy"]),
                                     min(fitted, key=lambda point: point["heldout"]["accuracy"]), history[-1])
    points = {name: {"step": point["step"], "error": 1 - point["heldout"]["accuracy"]}
              for name, point in (("initial", initial), ("first_minimum", minimum), ("peak", peak), ("final", final))}
    first_descent = points["initial"]["error"] - points["first_minimum"]["error"]
    rise = points["peak"]["error"] - points["first_minimum"]["error"]
    recovery = points["peak"]["error"] - points["final"]["error"]
    ordered = initial["step"] < minimum["step"] < peak["step"] < final["step"]
    return {**points, "first_descent": first_descent, "peak_rise": rise, "second_descent": recovery,
            "margin": margin, "full_error_double_descent": ordered and min(first_descent, rise, recovery) > margin,
            "selection": "minimum_before_sustained_train_fit; peak_while_train_fitted_before_heldout_target; "
                         "final_observation",
            "scope": "posthoc_synthetic_epoch_error_description; no_CIFAR_numerical_reproduction_or_causal_claim"}


def stretches(points, member):
    """The maximal runs of consecutive points satisfying `member`."""
    found, current = [], []
    for point in points:
        if member(point):
            current.append(point)
        else:
            if current:
                found.append(current)
            current = []
    if current:
        found.append(current)
    return found


def phases(history, target=.99, patience=2, ceiling=.1):
    """Sustained train fit, sustained held-out success, and the longest memorization plateau before it.

    A plateau is `patience` or more consecutive observations before the
    held-out onset with train accuracy at `target` or more and held-out
    accuracy at most `ceiling`. Grokking is observed when a plateau exists,
    the held-out onset follows the train onset, and the final held-out
    accuracy reaches `target`.
    """
    train = sustained(history, "train", target, patience)
    heldout = sustained(history, "heldout", target, patience)
    limit = heldout["onset"] if heldout else history[-1]["step"] + 1
    blocks = [block for block in stretches([point for point in history if point["step"] < limit], lambda point:
                                           point["train"]["accuracy"] >= target
                                           and point["heldout"]["accuracy"] <= ceiling)
              if len(block) >= patience]
    longest = max(blocks, key=lambda block: block[-1]["step"] - block[0]["step"]) if blocks else None
    plateau = {"start_step": longest[0]["step"], "end_step": longest[-1]["step"],
               "span_steps": longest[-1]["step"] - longest[0]["step"], "observations": len(longest),
               "minimum_train_accuracy": min(point["train"]["accuracy"] for point in longest),
               "maximum_heldout_accuracy": max(point["heldout"]["accuracy"] for point in longest)} if longest else None
    lag = heldout["onset"] - train["onset"] if train and heldout else None
    after = [point for point in history if heldout and point["step"] >= heldout["confirmed"]]
    final_target = history[-1]["heldout"]["accuracy"] >= target
    curve = error_curve(history, train, heldout, target)
    grokking = bool(plateau and lag is not None and lag > 0 and final_target)
    return {"sustained_train_fit": train, "sustained_heldout_target": heldout,
            "time_to_sustained_train_fit": times(history, train),
            "time_to_sustained_heldout_target": times(history, heldout),
            "lag_after_sustained_train_fit": lag, "memorization_plateau": plateau,
            "heldout_ceiling": ceiling, "train_target": target, "patience": patience,
            "fit_to_heldout_onset_ratio": heldout["onset"] / train["onset"]
            if train and heldout and train["onset"] else None,
            "observed_plateau_then_generalization": grokking, "epoch_error_curve_before_generalization": curve,
            "both_epoch_error_double_descent_and_grokking": bool(curve and curve["full_error_double_descent"]
                                                                 and grokking),
            "final_heldout_target": final_target, "observations_after_confirmation": len(after),
            "fraction_observations_at_target_after_confirmation":
                sum(point["heldout"]["accuracy"] >= target for point in after) / len(after) if after else None,
            "minimum_heldout_accuracy_after_confirmation":
                min(point["heldout"]["accuracy"] for point in after) if after else None,
            "scope": "scheduled_descriptive_diagnostic; near_chance_ceiling_is_explicit; no_causal_or_asymptotic_claim"}
