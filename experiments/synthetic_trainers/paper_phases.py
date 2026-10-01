"""Separate transient threshold crossings from a measured memorization phase.

Source: Convexifying Transformers, arXiv:2211.11052v1, Section 4's description
of fitting train before generalizing. The near-chance ceiling below is an
explicit diagnostic, not a recovered hyperparameter of that experiment.
"""


def sustained_onset(history, split, target, patience):
    for start in range(len(history) - patience + 1):
        window = history[start:start + patience]
        if all(point[split]["accuracy"] >= target for point in window):
            return {"onset": window[0]["step"], "confirmed": window[-1]["step"]}
    return None


def diagnose(report, heldout_ceiling=.1):
    config, history = report["plan"]["config"], report["history"]
    target, patience = config["target"], config["patience"]
    train = sustained_onset(history, "train", target, patience)
    heldout = sustained_onset(history, "heldout", target, patience)
    limit = heldout["onset"] if heldout else history[-1]["step"] + 1
    eligible = [point for point in history if point["step"] < limit]
    blocks, block = [], []
    for point in eligible:
        if point["train"]["accuracy"] >= target and point["heldout"]["accuracy"] <= heldout_ceiling:
            block.append(point)
        else:
            if len(block) >= patience:
                blocks.append(block)
            block = []
    if len(block) >= patience:
        blocks.append(block)
    longest = max(blocks, key=lambda block: block[-1]["step"] - block[0]["step"]) if blocks else None
    plateau = {"start_step": longest[0]["step"], "end_step": longest[-1]["step"],
               "span_steps": longest[-1]["step"] - longest[0]["step"], "observations": len(longest),
               "minimum_train_accuracy": min(point["train"]["accuracy"] for point in longest),
               "maximum_heldout_accuracy": max(point["heldout"]["accuracy"] for point in longest)} if longest else None
    lag = heldout["onset"] - train["onset"] if train and heldout else None
    after = [point for point in history if heldout and point["step"] >= heldout["confirmed"]]
    final_target = history[-1]["heldout"]["accuracy"] >= target
    return {"sustained_train_fit": train, "sustained_heldout_target": heldout,
            "lag_after_sustained_train_fit": lag,
            "memorization_plateau": plateau,
            "heldout_ceiling": heldout_ceiling, "train_target": target, "patience": patience,
            "fit_to_heldout_onset_ratio": heldout["onset"] / train["onset"] if train and heldout and train["onset"] else None,
            "observed_plateau_then_generalization": bool(plateau and lag is not None and lag > 0 and final_target),
            "final_heldout_target": final_target,
            "observations_after_confirmation": len(after),
            "fraction_observations_at_target_after_confirmation": sum(point["heldout"]["accuracy"] >= target for point in after) / len(after) if after else None,
            "minimum_heldout_accuracy_after_confirmation": min(point["heldout"]["accuracy"] for point in after) if after else None,
            "scope": "scheduled_descriptive_diagnostic; near_chance_ceiling_is_explicit; no_causal_or_asymptotic_claim"}
