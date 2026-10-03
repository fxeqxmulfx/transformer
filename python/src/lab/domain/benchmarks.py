"""Benchmarks: the task a model is trained and measured on."""

from dataclasses import dataclass
import math

from .spec import Spec, require


@dataclass(frozen=True)
class Benchmark(Spec, kind=True):
    """A task with fixed splits, a training loss and evaluation metrics."""

    @property
    def context(self):
        """Longest input the model reads."""
        raise NotImplementedError

    @property
    def observed(self):
        """The splits evaluated at every observation."""
        raise NotImplementedError


def is_prime(number):
    return number >= 2 and all(number % divisor for divisor in range(2, math.isqrt(number) + 1))


@dataclass(frozen=True)
class ModularDivision(Benchmark):
    """x / y mod p for every x and nonzero y, split once into train and held out.

    Source: Convexifying Transformers (arXiv:2211.11052v1), Section 4, in the
    token format of the openai/grok author code (commit 3d64b1d8): the row
    `<eos> x / y = q <eos>` supervises the answer and the final EOS. The split
    shuffles all p (p - 1) rows with the data seed and trains on the first
    round(fraction * rows).

    Batches walk shuffled epochs of the training rows. With tail "short" an
    epoch ends with its remainder as a smaller batch; with "wrap" a batch fills
    up from the next epoch.
    """
    prime: int
    train_fraction: float
    tail: str = "short"

    def check(self):
        require(is_prime(self.prime), "Division needs a prime modulus")
        require(0 < self.train_fraction < 1, "Train fraction lies strictly between zero and one")
        rows = self.prime * (self.prime - 1)
        require(0 < round(rows * self.train_fraction) < rows, "Both splits must be nonempty")
        require(self.tail in ("short", "wrap"), "Batch tail policy is 'short' or 'wrap'")

    @property
    def context(self):
        return 6

    @property
    def observed(self):
        return ("train", "heldout")
