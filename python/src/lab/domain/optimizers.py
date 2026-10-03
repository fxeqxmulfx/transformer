"""Optimizer blocks: the update rule an experiment trains under.

The adaptive rules follow their sources without bias correction, as
arXiv:1904.09237 and arXiv:1904.03590v4 state them; epsilon is a numerical
extension the sources do not have. Where a source lets a coefficient change
with the update t = 1, 2, ..., a `Decay` block multiplies it: `beta1_decay`
makes b1_t = b1 c_t, and `lr_decay` makes the step size alpha_t = lr c_t.
"""

from dataclasses import dataclass
import math

from .spec import Spec, require, require_kind


@dataclass(frozen=True)
class Decay(Spec, kind=True):
    """A multiplier c_t of update t = 1, 2, ..., with c_1 = 1."""


@dataclass(frozen=True)
class Constant(Decay):
    """c_t = 1."""


@dataclass(frozen=True)
class Inverse(Decay):
    """c_t = 1 / t."""


@dataclass(frozen=True)
class InverseSqrt(Decay):
    """c_t = 1 / sqrt(t)."""


@dataclass(frozen=True)
class Geometric(Decay):
    """c_t = ratio^(t - 1)."""
    ratio: float = 0.99

    def check(self):
        require(0 < self.ratio < 1, "A geometric ratio lies strictly between zero and one")


def check_rate(lr):
    require(math.isfinite(lr) and lr > 0, "Learning rate must be positive")


def check_decays(spec, *names):
    for name in names:
        require_kind(getattr(spec, name), Decay, name)


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
        check_rate(self.lr)
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
        check_rate(self.lr)
        require(self.weight_decay >= 0, "Decay must be nonnegative")


@dataclass(frozen=True)
class AMSGradW(Optimizer):
    """AMSGrad on raw moments, with decoupled decay.

    Source: arXiv:1904.03590v4, Algorithm 1 and Section 6, with the decay of
    arXiv:2606.25971v2, Sections 2 and 4.1; `Transformer.AMSGradW.trainingStep`.
    m <- b1_t m + (1 - b1_t) g, v <- b2 v + (1 - b2) g^2, vmax <- max(vmax, v),
    and x <- x - lr (c_t m / (sqrt(vmax) + eps) + weight_decay x), with c_t the
    `lr_decay`. Weight decay 0 is AMSGrad, `Transformer.AMSGrad.amsgradRule`;
    its Theorem 4.1 takes alpha_t = lr / sqrt(t) (`InverseSqrt()`) with
    b1_t = b1 / t (`Inverse()`) or b1 lambda^(t-1) (`Geometric(lambda)`).
    """
    lr: float
    betas: tuple[float, float]
    weight_decay: float
    eps: float = 1e-8
    beta1_decay: Decay = Constant()
    lr_decay: Decay = Constant()

    def check(self):
        check_rate(self.lr)
        require(len(self.betas) == 2 and 0 <= self.betas[0] < 1 and 0 <= self.betas[1] <= 1,
                "AMSGradW needs beta1 in [0, 1) and beta2 in [0, 1]")
        require(self.weight_decay >= 0 and self.eps > 0, "Decay must be nonnegative, epsilon positive")
        check_decays(self, "beta1_decay", "lr_decay")


@dataclass(frozen=True)
class Adam(Optimizer):
    """Adam without debiasing: AMSGrad's moments, normalized by v itself.

    Source: arXiv:1904.09237, Section 2, (Adam); `Transformer.AdamBeyond.adamRule`.
    x <- x - lr c_t m / (sqrt(v) + eps), with m and v as in `AMSGradW`.
    """
    lr: float
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    beta1_decay: Decay = Constant()
    lr_decay: Decay = Constant()

    def check(self):
        check_rate(self.lr)
        require(len(self.betas) == 2 and all(0 <= beta < 1 for beta in self.betas), "Adam needs two betas in [0, 1)")
        require(self.eps > 0, "Epsilon must be positive")
        check_decays(self, "beta1_decay", "lr_decay")


@dataclass(frozen=True)
class AdamX(Optimizer):
    """AMSGrad whose running maximum follows the change of b1_t.

    Source: arXiv:1904.03590v4, Section 5, Algorithm 2; `Transformer.AMSGrad.adamXRule`:
    vhat_t = max(((1 - b1_t) / (1 - b1_{t-1}))^2 vhat_{t-1}, v_t), vhat_1 = v_1,
    and x <- x - lr c_t m / (sqrt(vhat) + eps). Its Theorem 5.1 takes
    alpha_t = lr / sqrt(t) with any b1_t <= b1, and its Corollary 5.6
    b1_t = b1 lambda^(t-1) or b1 / t, the default.
    """
    lr: float
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    beta1_decay: Decay = Inverse()
    lr_decay: Decay = InverseSqrt()

    def check(self):
        check_rate(self.lr)
        require(len(self.betas) == 2 and all(0 <= beta < 1 for beta in self.betas), "AdamX needs two betas in [0, 1)")
        require(self.eps > 0, "Epsilon must be positive")
        check_decays(self, "beta1_decay", "lr_decay")


@dataclass(frozen=True)
class AdaGrad(Optimizer):
    """AdaGrad: the mean of the squared gradients at step size lr / sqrt(t).

    Source: arXiv:1904.09237, Section 2, (AdaGrad); `Transformer.AdamBeyond.adagradRule`:
    vhat <- ((t - 1) vhat + g^2) / t and x <- x - lr c_t g / (sqrt(vhat) + eps),
    so that the default c_t = 1 / sqrt(t) moves by lr g / sqrt(sum of g^2).
    """
    lr: float
    eps: float = 1e-8
    lr_decay: Decay = InverseSqrt()

    def check(self):
        check_rate(self.lr)
        require(self.eps > 0, "Epsilon must be positive")
        check_decays(self, "lr_decay")


@dataclass(frozen=True)
class AdamNC(Optimizer):
    """AdamNC with b2_t = 1 - 1/t: momentum over AdaGrad's mean of squared gradients.

    Source: arXiv:1904.09237, appendix, Algorithm 3, `Transformer.AdamBeyond.adamNCRule`:
    m <- b1_t m + (1 - b1_t) g, vhat as in `AdaGrad`, x <- x - lr c_t m / (sqrt(vhat) + eps).
    Its Theorem 5 takes alpha_t = lr / sqrt(t), and the remark after its
    Corollary 2 the decay b1_t = b1 / t; both are the defaults.
    """
    lr: float
    beta1: float = 0.9
    eps: float = 1e-8
    beta1_decay: Decay = Inverse()
    lr_decay: Decay = InverseSqrt()

    def check(self):
        check_rate(self.lr)
        require(0 <= self.beta1 < 1 and self.eps > 0, "AdamNC needs beta1 in [0, 1) and a positive epsilon")
        check_decays(self, "beta1_decay", "lr_decay")
