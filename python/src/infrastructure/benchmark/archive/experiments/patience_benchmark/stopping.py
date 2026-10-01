"""Validation-only stopping, separating exact best from meaningful progress."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class StopConfig:
    every: int = 250
    patience: int = 8
    min_delta: float = 0.0001
    min_steps: int = 1000
    max_steps: int = 20000
    divergence_delta: float = 0.1
    divergence_patience: int = 3

    def __post_init__(self):
        counts = (self.every, self.patience, self.max_steps, self.divergence_patience)
        if any(not isinstance(n, int) or n <= 0 for n in counts):
            raise ValueError("Check interval, patience and cap must be positive integers")
        if not isinstance(self.min_steps, int) or not 0 <= self.min_steps <= self.max_steps:
            raise ValueError("Invalid minimum update budget")
        if not math.isfinite(self.min_delta) or self.min_delta < 0:
            raise ValueError("min_delta must be finite and nonnegative")
        if not math.isfinite(self.divergence_delta) or self.divergence_delta <= 0:
            raise ValueError("divergence_delta must be finite and positive")


@dataclass(frozen=True)
class Decision:
    new_best: bool
    should_stop: bool
    reason: str | None = None


class EarlyStopper:
    def __init__(self, config):
        self.config = config
        self.best_loss = None
        self.best_step = None
        self.anchor_loss = None
        self.bad_checks = 0
        self.divergent_checks = 0
        self.last_step = -1

    def observe(self, step, validation_loss):
        if not isinstance(step, int) or step <= self.last_step:
            raise ValueError("Validation steps must strictly increase")
        self.last_step = step
        if not math.isfinite(validation_loss):
            return Decision(False, True, "nonfinite_validation")
        new_best = self.best_loss is None or validation_loss < self.best_loss
        if new_best:
            self.best_loss, self.best_step = validation_loss, step
        meaningful = self.anchor_loss is None or validation_loss < self.anchor_loss - self.config.min_delta
        if meaningful:
            self.anchor_loss, self.bad_checks = validation_loss, 0
        else:
            self.bad_checks += 1
        if validation_loss >= self.best_loss + self.config.divergence_delta:
            self.divergent_checks += 1
        else:
            self.divergent_checks = 0
        if step >= self.config.min_steps:
            if self.divergent_checks >= self.config.divergence_patience:
                return Decision(new_best, True, "validation_divergence")
            if self.bad_checks >= self.config.patience:
                return Decision(new_best, True, "patience")
        if step >= self.config.max_steps:
            return Decision(new_best, True, "max_steps")
        return Decision(new_best, False)


def choose_best_step(curves):
    valid = [row for row in curves if row["validation_loss"] is not None
             and math.isfinite(row["validation_loss"])]
    if not valid:
        raise ValueError("No finite validation observation")
    return min(valid, key=lambda row: (row["validation_loss"], row["step"]))["step"]
