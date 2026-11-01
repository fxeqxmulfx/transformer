"""Final-answer modular tasks for the normalized RoPE transformer."""

from dataclasses import dataclass

from .benchmarks import Benchmark, is_prime
from .spec import require


@dataclass(frozen=True)
class ModularPairs(Benchmark):
    """Exhaustive mixed arithmetic or powered-division control, raw residues.

    Source: review/flat_spectrum.md, Proposition 2 and Powered division
    (668989c4). Mixed labels are a^3+ab; division labels are (a/b)^exponent
    with b nonzero. Final-answer CE only, with the established SplitMix64
    half-table split. Fraction one is a fitting-only control. Unlike
    PowerDivision, this protocol reads a sequence model's last-position logits.
    """
    prime: int = 97
    operation: str = "mixed"
    exponent: int = 1
    train_fraction: float = .5

    def check(self):
        require(type(self.prime) is int and is_prime(self.prime), "ModularPairs needs a prime")
        require(self.operation in ("mixed", "division"), "Unknown modular pair operation")
        require(type(self.exponent) is int and self.exponent >= 0, "Exponent must be natural")
        require(self.operation == "division" or self.exponent == 1,
                "The mixed task has no exponent parameter; leave exponent at one")
        size = self.prime * (self.prime - (self.operation == "division"))
        require(0 < self.train_fraction <= 1 and round(size * self.train_fraction) > 0,
                "Training fraction must give a nonempty split")
        require(self.train_fraction == 1 or round(size * self.train_fraction) < size,
                "Partial-table training needs held-out rows")

    @property
    def context(self):
        return 2

    @property
    def observed(self):
        return ("train",) if self.train_fraction == 1 else ("train", "heldout")


@dataclass(frozen=True)
class ModularChain(Benchmark):
    """z <- z^3+b*z+c mod p for each input coefficient pair, answer only.

    Source: user-approved mixed-arithmetic progression, 2026-11-01;
    review/flat_spectrum.md, Proposition 2 covers each fixed-c local kernel,
    not the complete chain. Inputs are [z0,b0,c0,...]; no intermediate or
    answer token is supplied. Unique uniformly sampled inputs form fixed
    disjoint train/heldout sets. A separate longer split doubles the chain
    length and is evaluated only at the final checkpoint, without selection.
    Fresh uniform c avoids an absorbing-zero output bias. Counts are samples,
    not fractions of the exponentially large full domain.
    """
    prime: int = 97
    steps: int = 2
    train_examples: int = 4096
    heldout_examples: int = 2048
    longer_examples: int = 2048

    def check(self):
        require(type(self.prime) is int and is_prime(self.prime), "ModularChain needs a prime")
        for name in ("steps", "train_examples", "heldout_examples", "longer_examples"):
            require(type(getattr(self, name)) is int and getattr(self, name) > 0,
                    f"ModularChain {name} must be a positive integer")
        require(self.train_examples + self.heldout_examples <= self.prime ** (1 + 2 * self.steps),
                "Unique chain samples exceed the input domain")
        require(self.longer_examples <= self.prime ** (1 + 4 * self.steps),
                "Unique longer samples exceed their input domain")

    @property
    def context(self):
        return 1 + 4 * self.steps

    @property
    def observed(self):
        return ("train", "heldout")

    @property
    def last(self):
        return ("longer",)
