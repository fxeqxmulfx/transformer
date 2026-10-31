"""Current Rotate90 and powered division requested on 2026-10-31."""

from dataclasses import dataclass

from .benchmarks import Benchmark, is_prime
from .model import Model
from .spec import require


@dataclass(frozen=True)
class PeriRotate90(Model):
    """Tied E, quarter-turn operands and SwiGLU blocks between Peri-RMSNorms.

    Source: grokking_abmod/scripts/grok-simple.py, Rotate90, SHA256
    67c8c2f15d559180f8731e3a9efbc2759a51811c07e96932907334f1b0a38800;
    Transformer.Rotate90.Defs at 4a116463. The source's normal embedding
    initializer is used. Each block normalizes both its input and output;
    input and final norms are retained, without gains or biases. Hidden
    width defaults to (8*width+1)//3. No deviation from this forward pass.
    """
    width: int = 64
    depth: int = 1

    def check(self):
        require(type(self.width) is int and self.width >= 2 and self.width % 2 == 0,
                "PeriRotate90 needs a positive even width")
        require(type(self.depth) is int and self.depth >= 1,
                "PeriRotate90 needs a positive integer depth")

    @property
    def hidden(self):
        return (8 * self.width + 1) // 3

    @property
    def context(self):
        return 2


@dataclass(frozen=True)
class PowerDivision(Benchmark):
    """Every (a,b,(a/b)^exponent mod prime), with nonzero b and raw residues.

    Source: Transformer.Rotate90.existsUnique_div_pow at 1c9b23b2. The
    SplitMix64 split and short-tail sampler follow LabDivision. Extension:
    any natural power, including zero with 0^0=1; fraction one is an
    explicit full-table fitting control without a held-out split. No early
    stopping or model selection uses the reported held-out accuracies.
    """
    prime: int = 97
    exponent: int = 1
    train_fraction: float = 0.5

    def check(self):
        require(type(self.prime) is int and is_prime(self.prime), "PowerDivision needs a prime")
        require(type(self.exponent) is int and self.exponent >= 0,
                "PowerDivision exponent must be a natural number")
        require(0 < self.train_fraction <= 1, "Training fraction must lie in (0,1]")
        count = round(self.prime * (self.prime - 1) * self.train_fraction)
        require(count > 0, "The training split must be nonempty")
        require(self.train_fraction == 1 or count < self.prime * (self.prime - 1),
                "A partial-table run needs a held-out split")

    @property
    def context(self):
        return 2

    @property
    def observed(self):
        return ("train",) if self.train_fraction == 1 else ("train", "heldout")
