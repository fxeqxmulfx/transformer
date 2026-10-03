"""Retain fixed recovery windows throughout the observed generalization history.

These descriptive follow-ups to Convexifying Transformers, Section 4 supplement
the final-tail metrics. They preserve every canonical post-onset observation.
"""

from .persistence import PersistenceConfig
from .stability_recovery import describe


def timeline(report, criterion=PersistenceConfig(), *, window_steps=10000):
    description = describe(report, criterion, window_steps=window_steps)
    event = description["long_confirmation"]
    result = {"through_update": description["through_update"], "frozen_budget": description["frozen_budget"],
              "window_steps": window_steps, "long_confirmation": event,
              "scope": "fixed_grid_descriptive_post_long_onset_windows; partial_width_and_incomplete_windows_excluded_from_rates_and_changes"}
    if event is None:
        return {**result, "metrics": None}
    budget, cadence = report["plan"]["config"]["steps"], report["plan"]["config"]["eval_every"]
    first = min(((event["onset"] + window_steps - 1) // window_steps) * window_steps, budget)
    scheduled = sorted({0, budget, *range(cadence, budget + 1, cadence)})
    metrics = {}
    for name, score in (("joint", lambda p: min(p["train"]["accuracy"], p["heldout"]["accuracy"])),
                        ("heldout", lambda p: p["heldout"]["accuracy"])):
        episodes = description["metrics"][name]["episodes_after_long_onset"]

        def window(start, stop, nominal):
            belongs = lambda step: start <= step < stop or (stop == budget and step == stop)
            points = [p for p in report["history"] if belongs(p["step"])]
            expected = [step for step in scheduled if belongs(step)]
            failures = sum(score(p) < criterion.target for p in points)
            onsets = sum(belongs(e["first_failure"]) for e in episodes)
            complete = len(points) == len(expected)
            return {"start": start, "stop": stop, "stop_inclusive": stop == budget,
                    "nominal_width_window": nominal, "observations": len(points),
                    "expected_observations": len(expected), "complete_window": complete,
                    "failed_observations": failures, "failure_fraction": failures / len(points) if points else None,
                    "new_episode_onsets": onsets,
                    "episode_onsets_per_10000_updates": onsets * 10000 / (stop - start) if complete and nominal else None}

        leading = window(event["onset"], first, False) if first > event["onset"] else None
        windows = [window(start, min(start + window_steps, budget), start + window_steps <= budget)
                   for start in range(first, budget, window_steps)]
        changes = [{"previous_start": a["start"], "current_start": b["start"],
                    "failure_fraction_change": b["failure_fraction"] - a["failure_fraction"]}
                   for a, b in zip(windows, windows[1:])
                   if a["complete_window"] and b["complete_window"] and a["nominal_width_window"] and b["nominal_width_window"]]
        all_windows = ([leading] if leading else []) + windows
        assert sum(w["observations"] for w in all_windows) == description["metrics"][name]["post_long_onset_observations"]
        assert sum(w["failed_observations"] for w in all_windows) == description["metrics"][name]["post_long_onset_failed_observations"]
        metrics[name] = {"leading_partial_width_window": leading, "fixed_grid_windows": windows,
                         "adjacent_complete_nominal_window_changes": changes}
    return {**result, "metrics": metrics}
