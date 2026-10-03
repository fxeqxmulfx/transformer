"""Choosing a learning rate among runs that differ in nothing else.

A port of the selection of the convex MQAR comparison
(`experiments/convex_mqar/src/convex_mqar/validation_milestones.py`,
`build_report`). Policy "best" takes the run whose best observation ranks
highest; "first99" the run that first observed 99% accuracy on the selection
split, though it may have fallen after; "stable99" the earliest such
crossing among the runs that stayed at 99% through their last observation.
Crossings tie on the training seconds at the crossing, then on the smaller
rate; ranks tie on the smaller rate. Only the selection split is read,
never a final split.
"""

import json

POLICIES = ("best", "first99", "stable99")


def rate_groups(descriptions):
    """The labels of runs whose descriptions differ in `optimizer.lr` alone, in order; singletons are left out."""
    groups = {}
    for label, description in descriptions.items():
        optimizer = {key: value for key, value in description["optimizer"].items() if key != "lr"}
        groups.setdefault(json.dumps({**description, "optimizer": optimizer}, sort_keys=True), []).append(label)
    return [labels for labels in groups.values() if len(labels) > 1]


def choose(candidates, policy):
    """The candidate of one group that `policy` selects, or why none qualifies.

    A candidate is {"label", "lr", "rank", "crossing"}: the rank of its best
    observation, and its first observation at 99% ({"step",
    "training_seconds", "sustained_to_end"}), None if it never reached it.
    """
    if policy not in POLICIES:
        raise ValueError(f"Unknown rate selection policy: {policy}")
    if policy == "best":
        chosen = max(candidates, key=lambda candidate: (candidate["rank"], -candidate["lr"]))
    else:
        crossed = [candidate for candidate in candidates if candidate["crossing"] is not None]
        eligible = crossed if policy == "first99" else [
            candidate for candidate in crossed if candidate["crossing"]["sustained_to_end"]]
        if not eligible:
            return {"status": "99_percent_not_sustained" if crossed else "99_percent_unreached"}
        chosen = min(eligible, key=lambda candidate: (candidate["crossing"]["step"],
                                                      candidate["crossing"]["training_seconds"], candidate["lr"]))
    return {"status": "selected", "label": chosen["label"], "lr": chosen["lr"]}
