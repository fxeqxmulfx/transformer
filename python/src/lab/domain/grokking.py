"""Causal modular grokking indicators, with explicit study thresholds.

Source: Nanda et al., arXiv:2301.05217v1, section 5.1. Deviation: division
orbits replace selected addition frequencies; these thresholds are our
study choices. Signals use only past/current observations and do not
guarantee future success or change the optimization trajectory.
"""

from dataclasses import asdict, dataclass
import math
import statistics

from .spec import require
from .training import Diagnostics


@dataclass(frozen=True)
class GrokkingDiagnostics(Diagnostics):
    """Observe complete common-scaling orbits of actual division logits.

    Source: arXiv:2301.05217v1, section 5.1; our fixed division projection
    retains every invariant function, without future frequency selection.
    All vocabulary logits are kept. Zero quotient is excluded from the
    main energy fraction and retained in loss summaries. Probes preserve
    RNG/gradients and never update the optimizer. Norm sampling is inherited.
    """
    orbit_every: int = 1000
    batch: int = 512
    energy_floor: float = 1e-12

    def check(self):
        super().check()
        require(type(self.orbit_every) is int and self.orbit_every >= 1,
                "Orbit cadence must be a positive integer")
        require(type(self.batch) is int and self.batch >= 1, "Orbit batch must be positive")
        require(math.isfinite(self.energy_floor) and self.energy_floor > 0,
                "A positive finite energy floor rejects constant logits")


@dataclass(frozen=True)
class ProgressCriterion:
    """Initial causal detector, pinned separately from raw measurements."""
    target: float = .99
    patience: int = 5
    minimum_delay: int = 1000
    chance_ceiling: float = .1
    loss_window: int = 21
    smooth: int = 5
    log_drop: float = .1
    structure_window: int = 5
    structure_rise: float = .05
    cleanup_share: float = .9
    restricted_loss: float = .1

    def __post_init__(self):
        require(0 < self.target <= 1 and 0 <= self.chance_ceiling < self.target,
                "Accuracy thresholds must separate memorization from generalization")
        require(type(self.patience) is int and self.patience >= 2,
                "Confirmation needs at least two observations")
        require(type(self.minimum_delay) is int and self.minimum_delay > 0,
                "Delayed generalization needs a positive update delay")
        require(type(self.smooth) is int and self.smooth > 0
                and type(self.loss_window) is int and self.loss_window >= 2 * self.smooth,
                "Loss windows need disjoint smoothing intervals")
        require(type(self.structure_window) is int and self.structure_window >= 4,
                "Structure windows need disjoint pairs of observations")
        require(all(math.isfinite(x) and x > 0 for x in
                    (self.log_drop, self.structure_rise, self.cleanup_share, self.restricted_loss))
                and self.structure_rise <= 1 and self.cleanup_share <= 1,
                "Progress thresholds must be positive and finite")


def _accuracy(row, split):
    values = row[split]
    return values["answer_accuracy"] if "answer_accuracy" in values else values["accuracy"]


def _loss(row, split):
    values = row[split]
    return values["answer_loss"] if "answer_loss" in values else values["loss"]


def _drop(points, key, smooth):
    first = statistics.median(key(p) for p in points[:smooth])
    last = statistics.median(key(p) for p in points[-smooth:])
    return math.log(max(first, 1e-30) / max(last, 1e-30))


def available(history):
    """Older accuracy-only archives cannot support a loss-based indicator."""
    return bool(history) and all(
        split in row and any(key in row[split] for key in ("answer_accuracy", "accuracy"))
        and any(key in row[split] for key in ("answer_loss", "loss"))
        for row in history for split in ("train", "heldout"))


def norm_progress(records):
    """Actual sampled parameter/gradient/update norms, without a success claim.

    Source: arXiv:2301.05217v1, section 5.1 (weight norms). Deviation:
    group all actual optimizer tensors by module, retain gradients/updates
    and do not fit a grokking threshold to these scale-dependent quantities.
    """
    points = []
    for row in records:
        tensors = row.get("parameters", {})
        if not tensors or not all("parameter_l2" in value for value in tensors.values()):
            continue
        groups = {"all": list(tensors.values())}
        for group in ("embedding", "attention", "ffn"):
            groups[group] = [value for name, value in tensors.items() if
                             (name.startswith("embed.") if group == "embedding" else f".{group}." in name)]
        values = {}
        for group, entries in groups.items():
            values[group] = {key: math.sqrt(sum(entry[key] ** 2 for entry in entries))
                             for key in ("parameter_l2", "gradient_l2", "update_l2")
                             if entries and all(key in entry for entry in entries)}
            denominator = math.sqrt(sum(entry.get("parameter_before_l2", 0.) ** 2 for entry in entries))
            if denominator > 0 and "update_l2" in values[group]:
                values[group]["relative_update_l2"] = values[group]["update_l2"] / denominator
        points.append({"step": row["step"], "groups": values, "temperatures": row.get("temperatures", {})})
    return {"observations": points, "latest": points[-1] if points else None,
            "scope": "sampled_actual_optimizer_norms; scale_dependent; no_success_prediction"}


def progress(history, criterion=ProgressCriterion()):
    """Causal loss/structure evidence and observed delayed generalization.

    Loss-only alarms remain distinct from structure formation. An energy
    fraction alone does not certify correct answers. Every point is computed
    before reading the next row. Projection losses use the current model.
    """
    if not history:
        return {"criterion": asdict(criterion), "observations": []}
    points, structures, memory = [], [], []
    fitted, memorized, previous = None, False, -1
    for i, row in enumerate(history):
        step = row["step"]
        require(type(step) is int and step > previous,
                "Grokking observations must have strictly increasing integer updates")
        previous = step
        for split in ("train", "heldout"):
            require(0 <= _accuracy(row, split) <= 1, "Accuracy must be in [0, 1]")
            require(math.isfinite(_loss(row, split)) and _loss(row, split) >= 0,
                    "Grokking loss must be finite and nonnegative")
        recent = history[max(0, i + 1 - criterion.patience):i + 1]
        fit = len(recent) == criterion.patience and all(
            _accuracy(p, "train") >= criterion.target for p in recent)
        if fit and fitted is None:
            fitted = step
        if _accuracy(row, "train") >= criterion.target and _accuracy(row, "heldout") <= criterion.chance_ceiling:
            memory.append(row)
            if len(memory) >= criterion.patience and step - memory[0]["step"] >= criterion.minimum_delay:
                memorized = True
        else:
            memory = []
        lw = history[max(0, i + 1 - criterion.loss_window):i + 1]
        loss_drop = _drop(lw, lambda p: _loss(p, "heldout"), criterion.smooth) if len(lw) == criterion.loss_window else None
        eligible = fit and memorized and fitted is not None and step - fitted >= criterion.minimum_delay
        loss_alarm = bool(eligible and loss_drop is not None and loss_drop >= criterion.log_drop)
        if row.get("grokking") is not None:
            structures.append(row)
        sw = structures[-criterion.structure_window:]
        rise, restricted_drop, residual_drop = None, None, None
        forming, cleanup = False, False
        if len(sw) == criterion.structure_window and all(
                p["grokking"]["heldout_invariant_energy_fraction"] is not None for p in sw):
            rise = (statistics.median(p["grokking"]["heldout_invariant_energy_fraction"] for p in sw[-2:])
                    - statistics.median(p["grokking"]["heldout_invariant_energy_fraction"] for p in sw[:2]))
            restricted_drop = _drop(sw, lambda p: p["grokking"]["heldout_restricted_loss"], 2)
            residual_drop = _drop(sw, lambda p: p["grokking"]["heldout_residual_energy"], 2)
            forming = bool(eligible and rise >= criterion.structure_rise and restricted_drop >= criterion.log_drop)
            last = sw[-1]["grokking"]
            cleanup = bool(eligible and last["heldout_invariant_energy_fraction"] >= criterion.cleanup_share
                           and last["heldout_restricted_loss"] <= criterion.restricted_loss
                           and residual_drop >= criterion.log_drop)
        generalized = len(recent) == criterion.patience and all(
            min(_accuracy(p, "train"), _accuracy(p, "heldout")) >= criterion.target for p in recent)
        phase = ("generalized" if generalized else "cleanup" if cleanup else "forming" if forming
                 else "generalizing" if fit and _accuracy(row, "heldout") > criterion.chance_ceiling
                 else "memorizing" if fit else "fitting")
        points.append({"step": step, "phase": phase, "train_fit": fit, "observed_memorization": memorized,
                       "structure_step": structures[-1]["step"] if structures else None,
                       "heldout_log_loss_drop": loss_drop, "loss_only_alarm": loss_alarm,
                       "invariant_fraction_rise": rise, "restricted_log_loss_drop": restricted_drop,
                       "residual_log_energy_drop": residual_drop, "structure_forming": forming,
                       "cleanup": cleanup, "observed_delayed_generalization": memorized and generalized})
    events = {name: next((p["step"] for p in points if p[name]), None) for name in
              ("loss_only_alarm", "structure_forming", "cleanup", "observed_delayed_generalization")}
    return {"criterion": asdict(criterion), "first_events": events, "latest": points[-1],
            "observations": points, "structure_observations": len(structures),
            "loss_component": "answer_loss_when_available_else_recorded_combined_loss",
            "structure_component": "heldout_only_scaling_projection; training_logits_never_pooled",
            "scope": "causal_finite_observations; task_specific_structure; no_future_success_guarantee"}
