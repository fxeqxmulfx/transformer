"""Physical-weight decay compositions of magnitude/direction optimizers.

Source: local extensions of arXiv:2606.25971v2, Appendix A, Algorithm 2,
and arXiv:2405.15682v4, Section 4.4. Lean: Transformer.MDW.stepWith and
freeMDStep. The penalty acts on fused physical weights outside all moments.
Storage repair precedes latent averaging in the Schedule-Free variants.
"""

from dataclasses import dataclass
import math
from typing import ClassVar

from .optimizers import Optimizer, check_rate
from .spec import require


@dataclass(frozen=True)
class AMSGradMDW(Optimizer):
    """Raw-moment AMSGrad MD plus physical decay; Transformer.MDW.amsgradMDWProposal.

    The hidden matrices use the MD factorization, all other parameters use
    auxiliary AMSGradW. Every rate follows the outer schedule by default.
    The fused-weight penalty is the local addition to the cited MD source.
    """
    lr: float
    gain_rate: float = 1e-3
    auxiliary_rate: float = 3e-4
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.01
    schedule_components: bool = True
    allows_guard: ClassVar[bool] = False
    direction_stages: ClassVar[bool] = False

    def check(self):
        for rate in (self.lr, self.gain_rate, self.auxiliary_rate):
            check_rate(rate)
        require(len(self.betas) == 2 and 0 <= self.betas[0] < 1 and 0 <= self.betas[1] <= 1,
                "MDW needs beta1 in [0, 1) and beta2 in [0, 1]")
        require(math.isfinite(self.eps) and self.eps > 0, "Epsilon must be finite and positive")
        require(math.isfinite(self.weight_decay) and self.weight_decay >= 0,
                "Weight decay must be finite and nonnegative")
        require(type(self.schedule_components) is bool, "schedule_components must be a boolean")


@dataclass(frozen=True)
class AdamMDW(AMSGradMDW):
    """Bias-corrected Adam factors and auxiliary AdamW; Transformer.MDW.adamMDWProposal.

    Bias correction and epsilon follow native Adam. There is no running
    maximum. Projection and physical storage repair are the local MDW extension.
    """

    def check(self):
        super().check()
        require(self.betas[1] < 1, "AdamMDW needs beta2 in [0, 1)")


@dataclass(frozen=True)
class ScheduleFreeAMSGradMDW(AMSGradMDW):
    """Latent Schedule-Free AMSGrad MDW; Transformer.MDW.freeMDStep with maximum.

    Beta1 interpolates base and average, without a first-moment EMA. Raw
    epsilon placement, warmup and squared-rate weights follow Algorithm 1.
    The local addition decays and repairs the fused base before averaging.
    """
    warmup: int = 0

    def check(self):
        super().check()
        require(self.betas[1] < 1, "Schedule-Free MDW needs beta2 in [0, 1)")
        require(type(self.warmup) is int and self.warmup >= 0,
                "Built-in warmup must be a nonnegative integer")


@dataclass(frozen=True)
class ScheduleFreeAdamMDW(ScheduleFreeAMSGradMDW):
    """Raw-variance Schedule-Free Adam MDW; Transformer.MDW.freeMDStep without maximum.

    The local physical-decay composition uses Algorithm 1's variance instead
    of an AMSGrad maximum. Beta1 controls interpolation of latent factors.
    """
