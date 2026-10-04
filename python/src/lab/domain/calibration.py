"""Choosing a learning rate among runs that differ in nothing else.

A port of the selection of the convex MQAR comparison
(`experiments/convex_mqar/src/convex_mqar/validation_milestones.py`,
`build_report`). Policy "best" takes the run whose best observation ranks
highest; "first" the run that first reached the benchmark's target on the
selection split (`Benchmark.solved`), though it may have fallen after;
"stable" the earliest such crossing among the runs that stayed at the target
through their last observation. Crossings tie on the training seconds at
the crossing, then on the smaller rate; ranks tie on the smaller rate. Only
the selection split is read, never a final split. The comparison's target
was 99% accuracy, its policies "first99" and "stable99".
"""

import json

POLICIES = ("best", "first", "stable")


def rate_groups(descriptions):
    """The labels of runs whose descriptions differ in `optimizer.lr` alone, in order; singletons are left out."""
    groups = {}
    for label, description in descriptions.items():
        optimizer = {key: value for key, value in description["optimizer"].items() if key != "lr"}
        groups.setdefault(json.dumps({**description, "optimizer": optimizer}, sort_keys=True), []).append(label)
    return [labels for labels in groups.values() if len(labels) > 1]


def crossing(benchmark, history):
    """The first observation in `history` whose selection split reaches the benchmark's target, and whether every
    later one does too: {"step", "training_seconds", "sustained_to_end"}, None if none reaches it."""
    reached = [benchmark.solved(row[benchmark.selection]) for row in history]
    first = next((index for index, hit in enumerate(reached) if hit), None)
    if first is None:
        return None
    return {"step": history[first]["step"], "training_seconds": history[first]["training_seconds"],
            "sustained_to_end": all(reached[first:])}


def choose(candidates, policy):
    """The candidate of one group that `policy` selects, or why none qualifies.

    A candidate is {"label", "lr", "rank", "crossing"}: the rank of its best
    observation, and its `crossing` of the target, None if it never reached
    it.
    """
    if policy not in POLICIES:
        raise ValueError(f"Unknown rate selection policy: {policy}")
    if policy == "best":
        chosen = max(candidates, key=lambda candidate: (candidate["rank"], -candidate["lr"]))
    else:
        crossed = [candidate for candidate in candidates if candidate["crossing"] is not None]
        eligible = crossed if policy == "first" else [
            candidate for candidate in crossed if candidate["crossing"]["sustained_to_end"]]
        if not eligible:
            return {"status": "target_not_sustained" if crossed else "target_unreached"}
        chosen = min(eligible, key=lambda candidate: (candidate["crossing"]["step"],
                                                      candidate["crossing"]["training_seconds"], candidate["lr"]))
    return {"status": "selected", "label": chosen["label"], "lr": chosen["lr"]}
