"""Selection of the best observation, and early stopping on it."""

from dataclasses import dataclass
import math

from .spec import Spec, require


@dataclass(frozen=True)
class EarlyStopping(Spec):
    """Stop when the benchmark's selection loss stops improving or diverges.

    The policy of the historical text benchmarks (`gpt_mini.domain.stopping`):
    an observation makes progress when its loss lies more than `min_delta`
    below the last progress; `patience` observations without progress stop
    the run, and so do `divergence_patience` consecutive observations at least
    `divergence` above the best loss, both only from update `after` on. A
    nonfinite selection loss or gradient stops it at once.
    """
    patience: int = 8
    min_delta: float = 1e-4
    after: int = 1000
    divergence: float = 0.1
    divergence_patience: int = 3

    def check(self):
        require(self.patience >= 1 and self.divergence_patience >= 1, "Patience must be positive")
        require(self.after >= 0, "Stopping starts at a nonnegative update")
        require(math.isfinite(self.min_delta) and self.min_delta >= 0, "Progress margin must be nonnegative")
        require(math.isfinite(self.divergence) and self.divergence > 0, "Divergence margin must be positive")


class Selection:
    """The best observation of a selection loss so far, and why the run stops, if it does.

    A port of `gpt_mini.domain.stopping.EarlyStopper`. Without a policy only
    the best observation is kept; a nonfinite loss (None) is never the best.
    """

    def __init__(self, policy):
        self.policy = policy
        self.best = self.step = self.anchor = None
        self.bad = self.divergent = 0
        self.stop = None

    def observe(self, step, loss):
        """Whether the observation at `step` is the best so far; sets `stop` when the policy stops."""
        policy = self.policy
        if loss is None or not math.isfinite(loss):
            if policy is not None:
                self.stop = (step, "nonfinite_selection")
            return False
        new_best = self.best is None or loss < self.best
        if new_best:
            self.best, self.step = loss, step
        if policy is None:
            return new_best
        if self.anchor is None or loss < self.anchor - policy.min_delta:
            self.anchor, self.bad = loss, 0
        else:
            self.bad += 1
        self.divergent = self.divergent + 1 if loss >= self.best + policy.divergence else 0
        if step >= policy.after:
            if self.divergent >= policy.divergence_patience:
                self.stop = (step, "divergence")
            elif self.bad >= policy.patience:
                self.stop = (step, "patience")
        return new_best
