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
    """The best observation of the selection split so far, and why the run stops, if it does.

    A port of `gpt_mini.domain.stopping.EarlyStopper`, which kept the
    observation of the lowest loss. Here the best is the first observation of
    the highest `rank` of its metrics (`Benchmark.rank`), and the policy
    watches the loss. Without a policy only the best observation is kept; an
    observation whose loss is not finite (None) is never the best.
    """

    def __init__(self, policy, rank):
        self.policy, self.ranking = policy, rank
        self.rank = self.step = self.lowest = self.anchor = None
        self.bad = self.divergent = 0
        self.stop = None

    def observe(self, step, metrics):
        """Whether the observation at `step` is the best so far; sets `stop` when the policy stops."""
        policy, loss = self.policy, metrics["loss"]
        if loss is None or not math.isfinite(loss):
            if policy is not None:
                self.stop = (step, "nonfinite_selection")
            return False
        rank = self.ranking(metrics)
        new_best = self.rank is None or rank > self.rank
        if new_best:
            self.rank, self.step = rank, step
        if policy is None:
            return new_best
        self.lowest = loss if self.lowest is None else min(self.lowest, loss)
        if self.anchor is None or loss < self.anchor - policy.min_delta:
            self.anchor, self.bad = loss, 0
        else:
            self.bad += 1
        self.divergent = self.divergent + 1 if loss >= self.lowest + policy.divergence else 0
        if step >= policy.after:
            if self.divergent >= policy.divergence_patience:
                self.stop = (step, "divergence")
            elif self.bad >= policy.patience:
                self.stop = (step, "patience")
        return new_best
