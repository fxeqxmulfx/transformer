"""The browser Rotate90 model and matched analytic activation controls.

Source: grokking_abmod/web/own-channel/python/rotate90.py and README.md at
f49969393a7482490d45483d8a4fd3046b1adede; landscape_grokking_plan.md, M9.3.
This protocol is separate from openai/grok's EOS token format.
"""

from dataclasses import dataclass
import math

from .benchmarks import Benchmark, is_prime
from .model import Activation, Model, ReLU, SiLU
from .spec import Spec, require, require_kind


@dataclass(frozen=True)
class RotaryBranch(Spec, kind=True):
    """The residual branch of Rotate90 (research plan, M9.3)."""


@dataclass(frozen=True)
class DenseBranch(RotaryBranch):
    """r A followed by ReLU or SiLU; the browser uses ReLU.

    Source: the recorded browser README, Architecture and initialization.
    SiLU is the sole change in Neural.SiLU of the Lean framework.
    """
    activation: Activation

    def check(self):
        require_kind(self.activation, Activation, "activation")
        require(isinstance(self.activation, (ReLU, SiLU)), "A Rotate90 dense branch uses ReLU or SiLU")


@dataclass(frozen=True)
class SwiGLU(RotaryBranch):
    """(SiLU(r G) * (r U)) D, with `hidden` intermediate channels.

    Source: research plan, Activation comparisons, and Neural.SwiGLU in Lean.
    This changes the browser branch to three independent projections.
    G starts at variance 2/W, U at 1/W and D at 1/hidden; E is unchanged.
    """
    hidden: int

    def check(self):
        require(type(self.hidden) is int and self.hidden > 0, "SwiGLU hidden width must be a positive integer")


@dataclass(frozen=True)
class Rotate90(Model):
    """E[a] + J E[b], a pre-norm residual branch, final RMSNorm and tied E.T.

    Source: the recorded browser README and PyTorch export. DenseBranch uses
    its SplitMix64/Box-Muller initializer: E has variance 1/W, A has 2/W.
    No bias or normalization gain is learned. The vocabulary comes from the
    arithmetic benchmark. SwiGLU's extra matrices are specified above.
    """
    width: int
    branch: RotaryBranch
    eps: float = 1e-5

    def check(self):
        require(type(self.width) is int and self.width >= 2 and self.width % 2 == 0,
                "Rotate90 needs a positive even width of at least two")
        require_kind(self.branch, RotaryBranch, "branch")
        require(isinstance(self.eps, (int, float)) and math.isfinite(self.eps) and self.eps > 0,
                "Rotate90 normalization epsilon must be finite and positive")

    @property
    def context(self):
        return 2


@dataclass(frozen=True)
class LabDivision(Benchmark):
    """The browser's exhaustive (a, b, a/b mod p) rows without special tokens.

    Source: the recorded PyTorch export's dataset and Sampler. Rows start in
    numerator/denominator order, shuffle by SplitMix64 and split once. The
    browser uses p=97 and half the rows. Other primes and fractions extend
    that protocol; the train count is rounded to the nearest integer.
    """
    prime: int = 97
    train_fraction: float = 0.5

    def check(self):
        require(type(self.prime) is int and is_prime(self.prime), "LabDivision needs a prime modulus")
        require(0 < self.train_fraction < 1, "LabDivision train fraction lies between zero and one")
        rows = self.prime * (self.prime - 1)
        require(0 < round(rows * self.train_fraction) < rows, "Both arithmetic splits must be nonempty")

    @property
    def context(self):
        return 2

    @property
    def observed(self):
        return ("train", "heldout")
