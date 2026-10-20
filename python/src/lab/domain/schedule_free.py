"""Schedule-Free compositions and the authors' revised AdamW algorithm."""

from dataclasses import dataclass
import math
from typing import ClassVar

from .optimizers import Optimizer, check_rate
from .spec import require


@dataclass(frozen=True)
class ScheduleFreeAdamW(Optimizer):
    """Revised AdamW; Transformer.ScheduleFree.AdamW.trainingStep (e7ee4beb).

    Source: schedulefree 1.4.1, stability commit 59ed9b662d5ec5799aded7390b2482758423653c.
    Port: lean_optim.schedule_free_adamw.ScheduleFreeAdamW, with one reusable
    workspace and unchanged gradients. Bias correction divides the raw variance;
    epsilon is outside its corrected root. Decay acts at y. Averaging uses the
    squared largest rate so far (r=0, weight_lr_power=2), even at a zero rate
    after a positive one. Only z and v are stored. Beta1 interpolates iterates;
    eval() divides by it. Mode round trips can round in floating point.
    Rates follow the lab's schedule; a zero built-in warmup means none.
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
        require(len(self.betas) == 2 and 0 < self.betas[0] < 1 and 0 <= self.betas[1] < 1,
                "Schedule-Free AdamW needs beta1 in (0, 1) and beta2 in [0, 1)")
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
