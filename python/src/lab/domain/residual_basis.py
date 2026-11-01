"""Reserved-test Basis comparisons for the current normalized source GPT."""

from dataclasses import dataclass

from .spec import require
from .synthetic import Synthetic


@dataclass(frozen=True)
class ResidualBasis(Synthetic):
    """Task160: unchanged Basis data and sequence score, first99% endpoint.

    Source: domain.basis.BASIS and Synthetic, including free generation for
    parity and length128 validation for hard depth. Deviations: no best-state
    selection or automatic final-test evaluation; retain the current endpoint
    and open test only after the declared comparison closes. Task160 uses
    Schedule-Free AdamW and the source GPT, rather than historical GPTMini.
    """
    target: float = .99

    def check(self):
        super().check()
        require(self.study is None, "ResidualBasis preserves the original non-study splits")
        require(self.target == .99 and self.metric == "sequence_accuracy",
                "ResidualBasis stops at99% sequence accuracy")

    @property
    def validation_name(self):
        return "validation" if self.select is None else f"validation/{self.select}"

    @property
    def observed(self):
        return ("validation",) if self.select is None else ("validation", self.validation_name)

    @property
    def selection(self):
        return None

    @property
    def final(self):
        return ()

    @property
    def last(self):
        return ()

    def stable(self, history):
        return bool(history) and history[-1][self.validation_name][self.metric] >= self.target
