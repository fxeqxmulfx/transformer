"""Measure sampled failure frequency and recovery without changing the success gate.

Convexifying Transformers, Section 4 motivates delayed generalization. These
window/episode metrics are explicit descriptive follow-ups, not paper claims.
Canonical observations give no guarantee between samples or beyond the budget.
"""

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path

from .persistence import PersistenceConfig, assess


def _target_run(points, score, target):
    block = []
    for point in reversed(points):
        if score(point) < target:
            break
        block.append(point)
    return {"start": block[-1]["step"] if block else None,
            "end": block[0]["step"] if block else None, "observations": len(block),
            "sampled_span_steps": block[0]["step"] - block[-1]["step"] if block else 0}


def _episodes(points, score, target):
    episodes, opened = [], None
    for point in points:
        value = score(point)
        if value < target:
            if opened is None:
                opened = {"first_failure": point["step"], "failed_observations": 0,
                          "minimum_accuracy": value, "first_recovery": None}
            opened["last_failure"] = point["step"]
            opened["failed_observations"] += 1
            opened["minimum_accuracy"] = min(opened["minimum_accuracy"], value)
        elif opened is not None:
            opened["first_recovery"] = point["step"]
            episodes.append(opened)
            opened = None
    if opened is not None:
        episodes.append(opened)
    for episode in episodes:
        episode["maximum_threshold_deficit"] = target - episode["minimum_accuracy"]
        episode["failed_sample_span_steps"] = episode["last_failure"] - episode["first_failure"]
        recovery = episode["first_recovery"]
        episode["sampled_steps_until_first_recovery"] = recovery - episode["first_failure"] if recovery is not None else None
        episode["ongoing_at_last_observation"] = recovery is None
    return episodes


def describe(report, criterion=PersistenceConfig(), *, window_steps=10000):
    """Keep every fixed tail window, including empty and incomplete windows."""
    config, history = report["plan"]["config"], report["history"]
    budget, cadence = config["steps"], config["eval_every"]
    if (type(cadence) is not int or cadence <= 0 or type(budget) is not int or budget <= 0
            or type(window_steps) is not int or window_steps <= 0 or window_steps % cadence
            or criterion.tail_steps > budget or not history):
        raise ValueError("Recovery windows need a positive cadence multiple and a nonnegative tail start")
    end = history[-1]["step"]
    scheduled = sorted({0, budget, *range(cadence, budget + 1, cadence)})
    if ([p["step"] for p in history] != [step for step in scheduled if step <= end]
            or not 0 <= end <= budget or report["completed_steps"] != end
            or report["final"] != history[-1]
            or (report["plan"]["status"] == "complete" and end != budget)):
        raise ValueError("Recovery metrics require an exact ordered canonical prefix and matching final observation")
    for point in history:
        for split in ("train", "heldout"):
            value = point[split]["accuracy"]
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Recovery accuracy must be finite and between zero and one")
    assessment = assess(report, criterion)
    event = assessment["long_confirmation"]
    post = [p for p in history if event is not None and p["step"] >= event["onset"]]
    tail_start = budget - criterion.tail_steps
    metrics = {}
    for name, score in (("joint", lambda p: min(p["train"]["accuracy"], p["heldout"]["accuracy"])),
                        ("heldout", lambda p: p["heldout"]["accuracy"])):
        episodes = _episodes(post, score, criterion.target)
        windows = []
        for start in range(tail_start, budget, window_steps):
            stop = min(start + window_steps, budget)
            belongs = lambda step: start <= step < stop or (stop == budget and step == stop)
            expected = [step for step in scheduled if belongs(step)]
            points = [p for p in history if belongs(p["step"])]
            failures = sum(score(p) < criterion.target for p in points)
            complete = len(points) == len(expected)
            onsets = sum(belongs(p["first_failure"]) for p in episodes) if event else None
            windows.append({"start": start, "stop": stop, "stop_inclusive": stop == budget,
                            "observations": len(points), "expected_observations": len(expected),
                            "complete_window": complete, "failed_observations": failures,
                            "failure_fraction": failures / len(points) if points else None,
                            "new_episode_onsets": onsets,
                            "episode_onsets_per_10000_updates": onsets * 10000 / (stop - start)
                            if complete and onsets is not None else None})
        changes = [{"previous_start": a["start"], "current_start": b["start"],
                    "failure_fraction_change": b["failure_fraction"] - a["failure_fraction"]}
                   for a, b in zip(windows, windows[1:]) if a["complete_window"] and b["complete_window"]]
        failed = [p for p in history if score(p) < criterion.target]
        post_failures = sum(score(p) < criterion.target for p in post)
        metrics[name] = {"post_long_onset_observations": len(post),
                         "post_long_onset_failed_observations": post_failures,
                         "post_long_onset_failure_fraction": post_failures / len(post) if post else None,
                         "episodes_after_long_onset": episodes, "episode_count": len(episodes) if event else None,
                         "last_failed_observation": failed[-1]["step"] if failed else None,
                         "updates_since_last_failed_observation": end - failed[-1]["step"] if failed else None,
                         "final_target_streak": _target_run(history, score, criterion.target),
                         "fixed_tail_windows": windows, "adjacent_complete_window_changes": changes}
    return {"through_update": end, "frozen_budget": budget, "window_steps": window_steps,
            "criterion": asdict(criterion), "complete_budget": assessment["complete_canonical_history"],
            "long_confirmation": event, "frozen_persistent_final_performance": assessment["persistent_final_performance"],
            "frozen_stable_grokking": assessment["stable_grokking"], "metrics": metrics,
            "scope": "descriptive_canonical_frequency_and_recovery; no_changed_gate_or_monotone_statistical_or_future_stability_claim"}


if __name__ == "__main__":
    from .stability_report import verify_archive
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--window-steps", type=int, default=10000)
    args = parser.parse_args()
    verify_archive(args.archive)
    plan = json.loads((args.archive / "plan.json").read_text())
    report = json.loads((args.archive / "measurements.json").read_text())
    print(json.dumps(describe(report, PersistenceConfig(**plan["criterion"]), window_steps=args.window_steps)))
