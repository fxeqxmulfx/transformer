"""Whether a modular run's success persisted, and how it failed and recovered, read from its observations.

Ports of the stability analyses of `experiments/synthetic_trainers`:
`persistence.assess` (`persistence`) and `stability_recovery.describe`
(`recovery`). The setting is the delayed generalization of Convexifying
Transformers (arXiv:2211.11052v1), Section 4; the criterion, its spans and
supports and the recovery windows are choices of the follow-up study, not
the paper's. They describe a measured history and claim no cause.

The canonical observations of a budget are update 0, every `every` updates
and the last update. A run is complete when it trained its whole budget and
observed exactly those updates; nothing is claimed between observations or
beyond the budget. The phases of the run are read by `phases.phases`, beside
the persistence, where the historical assessment embedded them.
"""

from dataclasses import asdict, dataclass
import itertools
import math

from .phases import stretches
from .spec import require


@dataclass(frozen=True)
class Persistence:
    """The success criterion the stability protocols fixed before training (`PersistenceConfig`).

    An observation succeeds when train and held-out accuracy both reach
    `target`. A plateau needs `plateau_observations` observations spanning
    `plateau_steps` updates with train fitted and held-out at most
    `heldout_ceiling`; a long confirmation is `confirmation_observations`
    consecutive successes; the tail is the last `tail_steps` updates.
    """
    target: float = .99
    heldout_ceiling: float = .1
    plateau_steps: int = 1000
    plateau_observations: int = 5
    confirmation_observations: int = 20
    tail_steps: int = 50000

    def __post_init__(self):
        require(0 < self.target <= 1 and 0 <= self.heldout_ceiling < self.target, "Invalid accuracy thresholds")
        require(min(self.plateau_steps, self.tail_steps, self.plateau_observations,
                    self.confirmation_observations) >= 1, "Persistence spans and supports must be positive")

    def success(self, point):
        return min(point["train"]["accuracy"], point["heldout"]["accuracy"]) >= self.target


CRITERION = Persistence()


def observed_updates(budget, every):
    """The canonical observations of a budget: update 0, every multiple of `every`, and the last update."""
    return sorted({0, budget, *range(every, budget + 1, every)})


def persistence(history, budget, every, finished, criterion=CRITERION):
    """The long confirmation, the plateau before the held-out onset, and whether the whole tail succeeded.

    Final performance persisted when the run is complete and every
    observation of the tail succeeded; grokking was stable when it persisted
    and its long confirmation began after a plateau.
    """
    if not history:
        raise ValueError("Persistence assessment needs a measured history")
    complete = finished and [point["step"] for point in history] == observed_updates(budget, every)
    eligible = [point for point in history if point["step"] <= budget]
    event = None
    for start in range(len(eligible) - criterion.confirmation_observations + 1):
        window = eligible[start:start + criterion.confirmation_observations]
        if all(criterion.success(point) for point in window):
            event = {"onset": window[0]["step"], "confirmed": window[-1]["step"],
                     "training_seconds": window[-1].get("training_seconds"),
                     "wall_seconds": window[-1].get("wall_seconds")}
            break
    onset = next((point["step"] for point in eligible if point["heldout"]["accuracy"] >= criterion.target),
                 budget + 1)
    plateaus = [block for block in stretches(eligible, lambda point: point["step"] < onset
                                             and point["train"]["accuracy"] >= criterion.target
                                             and point["heldout"]["accuracy"] <= criterion.heldout_ceiling)
                if len(block) >= criterion.plateau_observations
                and block[-1]["step"] - block[0]["step"] >= criterion.plateau_steps]
    tail_start = budget - criterion.tail_steps
    tail = [point for point in eligible if point["step"] >= tail_start]
    tail_pass = bool(complete and tail_start >= 0 and tail and all(criterion.success(point) for point in tail))
    plateau = max(plateaus, key=lambda block: block[-1]["step"] - block[0]["step"]) if plateaus else None
    return {"criterion": asdict(criterion), "complete_canonical_history": complete,
            "plateau": {"start": plateau[0]["step"], "end": plateau[-1]["step"],
                        "observations": len(plateau)} if plateau else None,
            "long_confirmation": event, "tail_start_step": tail_start, "tail_observations": len(tail),
            "tail_failures": [point["step"] for point in tail if not criterion.success(point)],
            "tail_minimum_heldout_accuracy": min(point["heldout"]["accuracy"] for point in tail) if tail else None,
            "persistent_final_performance": tail_pass,
            "stable_grokking": bool(tail_pass and plateau and event and event["onset"] > plateau[-1]["step"]),
            "scope": "finite_scheduled_observations; no_guarantee_between_observations_or_beyond_budget"}


def streak(points, score, target):
    """The run of observations at `target` or more that ends the history."""
    block = []
    for point in reversed(points):
        if score(point) < target:
            break
        block.append(point)
    return {"start": block[-1]["step"] if block else None, "end": block[0]["step"] if block else None,
            "observations": len(block), "sampled_span_steps": block[0]["step"] - block[-1]["step"] if block else 0}


def episodes(points, score, target):
    """Each run of observations below `target`: its first and last failure, its depth and its recovery."""
    found = []
    for block in stretches(points, lambda point: score(point) < target):
        recovered = next((point["step"] for point in points if point["step"] > block[-1]["step"]), None)
        minimum = min(score(point) for point in block)
        found.append({"first_failure": block[0]["step"], "failed_observations": len(block),
                      "minimum_accuracy": minimum, "first_recovery": recovered, "last_failure": block[-1]["step"],
                      "maximum_threshold_deficit": target - minimum,
                      "failed_sample_span_steps": block[-1]["step"] - block[0]["step"],
                      "sampled_steps_until_first_recovery": None if recovered is None else recovered - block[0]["step"],
                      "ongoing_at_last_observation": recovered is None})
    return found


def recovery(history, budget, every, finished, criterion=CRITERION, window=10000):
    """Failures after the long confirmation began, as episodes and as fixed windows over the tail.

    The history may stop short of the budget, but it must be every canonical
    observation up to its last. Each failure measure is read twice: for the
    joint success of train and held-out, and for held-out alone.
    """
    if window <= 0 or window % every or criterion.tail_steps > budget or not history:
        raise ValueError("Recovery windows need a positive cadence multiple and a nonnegative tail start")
    end = history[-1]["step"]
    scheduled = observed_updates(budget, every)
    if ([point["step"] for point in history] != [step for step in scheduled if step <= end]
            or (finished and end != budget)):
        raise ValueError("Recovery metrics require an exact ordered canonical prefix")
    if not all(math.isfinite(point[split]["accuracy"]) and 0 <= point[split]["accuracy"] <= 1
               for point in history for split in ("train", "heldout")):
        raise ValueError("Recovery accuracy must be finite and between zero and one")
    assessment = persistence(history, budget, every, finished, criterion)
    event = assessment["long_confirmation"]
    post = [point for point in history if event is not None and point["step"] >= event["onset"]]
    tail_start = budget - criterion.tail_steps
    metrics = {}
    for name, score in (("joint", lambda point: min(point["train"]["accuracy"], point["heldout"]["accuracy"])),
                        ("heldout", lambda point: point["heldout"]["accuracy"])):
        found = episodes(post, score, criterion.target)
        windows = []
        for start in range(tail_start, budget, window):
            stop = min(start + window, budget)
            inside = [step for step in scheduled if start <= step < stop or step == stop == budget]
            points = [point for point in history if point["step"] in inside]
            failures = sum(score(point) < criterion.target for point in points)
            complete = len(points) == len(inside)
            onsets = sum(episode["first_failure"] in inside for episode in found) if event else None
            windows.append({"start": start, "stop": stop, "stop_inclusive": stop == budget,
                            "observations": len(points), "expected_observations": len(inside),
                            "complete_window": complete, "failed_observations": failures,
                            "failure_fraction": failures / len(points) if points else None,
                            "new_episode_onsets": onsets,
                            "episode_onsets_per_10000_updates": onsets * 10000 / (stop - start)
                            if complete and onsets is not None else None})
        changes = [{"previous_start": a["start"], "current_start": b["start"],
                    "failure_fraction_change": b["failure_fraction"] - a["failure_fraction"]}
                   for a, b in itertools.pairwise(windows) if a["complete_window"] and b["complete_window"]]
        failed = [point for point in history if score(point) < criterion.target]
        post_failures = sum(score(point) < criterion.target for point in post)
        metrics[name] = {"post_long_onset_observations": len(post),
                         "post_long_onset_failed_observations": post_failures,
                         "post_long_onset_failure_fraction": post_failures / len(post) if post else None,
                         "episodes_after_long_onset": found, "episode_count": len(found) if event else None,
                         "last_failed_observation": failed[-1]["step"] if failed else None,
                         "updates_since_last_failed_observation": end - failed[-1]["step"] if failed else None,
                         "final_target_streak": streak(history, score, criterion.target),
                         "fixed_tail_windows": windows, "adjacent_complete_window_changes": changes}
    return {"through_update": end, "frozen_budget": budget, "window_steps": window, "criterion": asdict(criterion),
            "complete_budget": assessment["complete_canonical_history"], "long_confirmation": event,
            "frozen_persistent_final_performance": assessment["persistent_final_performance"],
            "frozen_stable_grokking": assessment["stable_grokking"], "metrics": metrics,
            "scope": "descriptive_canonical_frequency_and_recovery; "
                     "no_changed_gate_or_monotone_statistical_or_future_stability_claim"}
