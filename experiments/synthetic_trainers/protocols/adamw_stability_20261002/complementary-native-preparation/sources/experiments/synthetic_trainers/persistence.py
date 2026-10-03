"""Prospectively specified modular phase and final persistence criteria.

Source setting: Convexifying Transformers, Section 4. The lengths, thresholds
and exhaustive tail checks below are explicit follow-up study choices.
"""

from dataclasses import asdict, dataclass

from .paper_phases import diagnose


@dataclass(frozen=True)
class PersistenceConfig:
    target: float = .99
    heldout_ceiling: float = .1
    plateau_steps: int = 1000
    plateau_observations: int = 5
    confirmation_observations: int = 20
    tail_steps: int = 50000

    def __post_init__(self):
        if not 0 < self.target <= 1 or not 0 <= self.heldout_ceiling < self.target:
            raise ValueError("Invalid accuracy thresholds")
        if min(self.plateau_steps, self.tail_steps, self.plateau_observations,
               self.confirmation_observations) < 1:
            raise ValueError("Persistence spans and supports must be positive")


def assess(report, criterion=PersistenceConfig()):
    config, history = report["plan"]["config"], report["history"]
    if not history:
        raise ValueError("Persistence assessment needs a measured history")
    budget, cadence = config["steps"], config["eval_every"]
    expected = sorted({0, budget, *range(cadence, budget + 1, cadence)})
    complete = (report["plan"]["status"] == "complete" and report["completed_steps"] == budget
                and [point["step"] for point in history] == expected
                and report["final"] == history[-1])
    eligible = [point for point in history if point["step"] <= budget]
    success = lambda p: min(p["train"]["accuracy"], p["heldout"]["accuracy"]) >= criterion.target
    event = None
    for start in range(len(eligible) - criterion.confirmation_observations + 1):
        window = eligible[start:start + criterion.confirmation_observations]
        if all(success(point) for point in window):
            event = {"onset": window[0]["step"], "confirmed": window[-1]["step"],
                     "training_seconds": window[-1].get("training_seconds"),
                     "wall_seconds": window[-1].get("wall_seconds")}
            break
    onset = next((p["step"] for p in eligible if p["heldout"]["accuracy"] >= criterion.target), budget + 1)
    blocks, block = [], []
    for point in eligible:
        if (point["step"] < onset and point["train"]["accuracy"] >= criterion.target
                and point["heldout"]["accuracy"] <= criterion.heldout_ceiling):
            block.append(point)
        else:
            if block:
                blocks.append(block)
            block = []
    if block:
        blocks.append(block)
    plateaus = [b for b in blocks if len(b) >= criterion.plateau_observations
                and b[-1]["step"] - b[0]["step"] >= criterion.plateau_steps]
    tail_start = budget - criterion.tail_steps
    tail = [point for point in eligible if point["step"] >= tail_start]
    tail_pass = bool(complete and tail_start >= 0 and tail and all(success(p) for p in tail))
    plateau = max(plateaus, key=lambda b: b[-1]["step"] - b[0]["step"]) if plateaus else None
    return {"criterion": asdict(criterion), "complete_canonical_history": complete,
            "plateau": {"start": plateau[0]["step"], "end": plateau[-1]["step"],
                        "observations": len(plateau)} if plateau else None,
            "long_confirmation": event, "tail_start_step": tail_start,
            "tail_observations": len(tail),
            "tail_failures": [p["step"] for p in tail if not success(p)],
            "tail_minimum_heldout_accuracy": min(p["heldout"]["accuracy"] for p in tail) if tail else None,
            "persistent_final_performance": tail_pass,
            "stable_grokking": bool(tail_pass and plateau and event and event["onset"] > plateau[-1]["step"]),
            "legacy_phase_diagnostics": diagnose(report),
            "scope": "finite_scheduled_observations; no_guarantee_between_observations_or_beyond_budget"}
