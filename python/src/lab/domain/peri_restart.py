"""Controlled restarts for the Rotate90 learning-gap investigation."""

from dataclasses import dataclass
import math

from .model import Model
from .peri_rotary import PeriRotate90
from .spec import require


@dataclass(frozen=True)
class PeriRotate90Restart(Model):
    """Initialize the unchanged forward from a pinned training-query checkpoint.

    Source: review/flat_spectrum.md, Alias symmetry and training; the
    PeriRotate90 forward at a7c012ea. Deviations concern initialization and
    trainable parameters only: optional subgroup averaging, relative Gaussian
    perturbations, and freezing either the embedding or the branch matrices.
    Optimizer histories are deliberately reset in every restart control.
    """
    network: PeriRotate90
    checkpoint: str
    sha256: str
    alias_size: int = 1
    noise: float = 0.
    train_embedding: bool = True
    train_branch: bool = True

    def check(self):
        require(isinstance(self.network, PeriRotate90), "Restart needs the current PeriRotate90")
        self.network.check()
        require(isinstance(self.checkpoint, str) and bool(self.checkpoint), "Checkpoint path is required")
        require(isinstance(self.sha256, str) and len(self.sha256) == 64 and
                all(c in "0123456789abcdef" for c in self.sha256), "Checkpoint SHA256 is required")
        require(type(self.alias_size) is int and self.alias_size >= 1, "Alias size must be positive")
        require(isinstance(self.noise, (int, float)) and not isinstance(self.noise, bool) and
                math.isfinite(self.noise) and self.noise >= 0, "Relative perturbation must be nonnegative")
        require(type(self.train_embedding) is bool and type(self.train_branch) is bool,
                "Trainable selections must be boolean")
        require(self.train_embedding or self.train_branch, "At least one parameter family must train")

    @property
    def context(self):
        return self.network.context

    @property
    def width(self):
        return self.network.width

    @property
    def depth(self):
        return self.network.depth

    @property
    def hidden(self):
        return self.network.hidden
