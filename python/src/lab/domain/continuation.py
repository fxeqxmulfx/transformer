"""Preserve Schedule-Free histories in matched rate/decay continuations."""

from dataclasses import dataclass

from .spec import require
from .training import CheckpointHistory


@dataclass(frozen=True)
class ScheduleFreeContinuation(CheckpointHistory):
    """Continue a pinned query, base, second moment, averaging clock and sampler.

    Source: review/flat_spectrum.md, Gradient forcing, F1--F3. Only the new
    recipe's rate and decay override the source optimizer settings; beta,
    epsilon, warmup, model, task, seeds and batch size must agree. Budgets and
    recorded update numbers remain absolute. The initial observation and
    compilation preserve the source query exactly, avoiding extra mode-roundtrip
    rounding before the first continuation step. Later observations use the
    ordinary engine unchanged. Every subsequent checkpoint is retained.
    """
    source: str = ""
    sha256: str = ""

    def check(self):
        super().check()
        require(isinstance(self.source, str) and bool(self.source), "Source checkpoint is required")
        require(isinstance(self.sha256, str) and len(self.sha256) == 64 and
                all(c in "0123456789abcdef" for c in self.sha256), "Source SHA256 is required")
