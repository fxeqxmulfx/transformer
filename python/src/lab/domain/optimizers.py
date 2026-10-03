"""Optimizer blocks: the update rule an experiment trains under."""

from dataclasses import dataclass
import math

from .spec import Spec, require


@dataclass(frozen=True)
class Optimizer(Spec, kind=True):
    """The update rule."""


@dataclass(frozen=True)
class AdamW(Optimizer):
    """torch.optim.AdamW: bias-corrected moments and decoupled decay.

    `decay` selects the decayed tensors: "all" trainable parameters, or only
    the "matrices" (ndim >= 2) with vectors in an undecayed group.
    """
    lr: float
    betas: tuple[float, float]
    weight_decay: float
    eps: float = 1e-8
    decay: str = "all"

    def check(self):
        require(math.isfinite(self.lr) and self.lr > 0, "Learning rate must be positive")
        require(len(self.betas) == 2 and all(0 <= beta < 1 for beta in self.betas),
                "AdamW needs two betas in [0, 1)")
        require(self.weight_decay >= 0 and self.eps > 0, "Decay must be nonnegative, epsilon positive")
        require(self.decay in ("all", "matrices"), "AdamW decay scope is 'all' or 'matrices'")


@dataclass(frozen=True)
class SGD(Optimizer):
    """Gradient descent with decoupled decay: x <- x - lr (g + weight_decay x)."""
    lr: float
    weight_decay: float = 0.0

    def check(self):
        require(math.isfinite(self.lr) and self.lr > 0, "Learning rate must be positive")
        require(self.weight_decay >= 0, "Decay must be nonnegative")


@dataclass(frozen=True)
class AMSGradW(Optimizer):
    """AMSGrad on raw moments, with decoupled decay.

    Source: arXiv:1904.03590v4, Algorithm 1 and Section 6, with the decay of
    arXiv:2606.25971v2, Sections 2 and 4.1; `Transformer.AMSGradW.trainingStep`.
    No bias correction: m <- b1 m + (1 - b1) g, v <- b2 v + (1 - b2) g^2,
    vmax <- max(vmax, v), and x <- x - lr (m / (sqrt(vmax) + eps) + weight_decay x).
    Weight decay 0 is AMSGrad.
    """
    lr: float
    betas: tuple[float, float]
    weight_decay: float
    eps: float = 1e-8

    def check(self):
        require(math.isfinite(self.lr) and self.lr > 0, "Learning rate must be positive")
        require(len(self.betas) == 2 and 0 <= self.betas[0] < 1 and 0 <= self.betas[1] <= 1,
                "AMSGradW needs beta1 in [0, 1) and beta2 in [0, 1]")
        require(self.weight_decay >= 0 and self.eps > 0, "Decay must be nonnegative, epsilon positive")
