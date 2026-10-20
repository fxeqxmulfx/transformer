"""Schedule-Free compositions and the paper's specialized AdamW algorithm."""

from dataclasses import dataclass
import math
from typing import ClassVar

from .optimizers import Optimizer, check_rate
from .spec import require


@dataclass(frozen=True)
class ScheduleFreeAdamW(Optimizer):
    """Query decay and squared effective-rate averages; Transformer.ScheduleFree.adamStep.

    Port: lean_optim.schedule_free.ScheduleFreeAdamW at 4f764a4d, with stable
    interpolation and one denominator workspace. Beta1 interpolates iterates;
    there is no first-moment EMA. Epsilon is added to the raw variance root.
    A zero warmup means none, completing the source's undefined division by zero.
    Rates follow the lab's schedule; constant rates after warmup match the paper.
    Evaluation uses the averaged weights for both training and held-out data.
    """
    lr: float
    betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 0.0
    eps: float = 1e-8
    warmup: int = 0
    allows_guard: ClassVar[bool] = False
    direction_stages: ClassVar[bool] = False

    def check(self):
        check_rate(self.lr)
        require(len(self.betas) == 2 and all(0 <= beta < 1 for beta in self.betas),
                "Schedule-Free AdamW needs two betas in [0, 1)")
        require(math.isfinite(self.eps) and self.eps > 0, "Epsilon must be finite and positive")
        require(math.isfinite(self.weight_decay) and self.weight_decay >= 0,
                "Weight decay must be finite and nonnegative")
        require(type(self.warmup) is int and self.warmup >= 0,
                "Built-in warmup must be a nonnegative integer")


@dataclass(frozen=True)
class ScheduleFreeMuon(Optimizer):
    """General Schedule-Free wrapper around native Muon; equal prefix averages.

    Sources: arXiv:2405.15682v4, Sections 2-2.1, and arXiv:2502.16982,
    Sections 2.1-2.2. Lean: Transformer.ScheduleFree.muonWrapperRun.
    This composition keeps Muon's native momentum and finite Newton-Schulz
    direction. Beta controls outer interpolation independently. Decay acts
    on the base point. Auxiliary Adam blocks share the same wrapper.
    The initial point has weight one; optional warmup is linear. The lab's
    external schedule may vary rates; constant rates after warmup need none.
    Neither source provides a convergence theorem for this composition.
    """
    lr: float
    beta: float = 0.9
    momentum: float = 0.95
    auxiliary: float = 0.05
    weight_decay: float = 0.0
    warmup: int = 0
    allows_guard: ClassVar[bool] = False
    direction_stages: ClassVar[bool] = False

    def check(self):
        check_rate(self.lr)
        require(0 <= self.beta <= 1, "Schedule-Free beta must lie in [0, 1]")
        require(0 <= self.momentum < 1, "Muon momentum must lie in [0, 1)")
        require(math.isfinite(self.auxiliary) and self.auxiliary > 0,
                "Auxiliary Adam multiplier must be finite and positive")
        require(math.isfinite(self.weight_decay) and self.weight_decay >= 0,
                "Weight decay must be finite and nonnegative")
        require(type(self.warmup) is int and self.warmup >= 0,
                "Built-in warmup must be a nonnegative integer")
