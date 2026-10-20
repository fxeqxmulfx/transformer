"""Muon in the general Schedule-Free scheme with explicit evaluation buffers.

Composition: arXiv:2405.15682v4, Sections 2-2.1, and arXiv:2502.16982,
Sections 2.1-2.2. Lean: Transformer.ScheduleFree.muonWrapperRun.
The base optimizer retains native Muon Nesterov momentum, five finite
Newton-Schulz steps and shape scaling. Nonselected parameters use the
existing Muon auxiliary bias-corrected Adam direction. Every block uses
the same mixed query and equal averaging, including the initial point.
This composition has no nonconvex convergence guarantee from either paper.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ._parameters import Options, Parameters, settings
from .matrix import muon_direction
from .schedule_free import ScheduleFreeSGD

if TYPE_CHECKING:
    from collections.abc import Iterable

    from torch import Tensor

    from ._types import Closure, Loss, Params


class ScheduleFreeMuon(ScheduleFreeSGD):
    """Wrap native Muon; beta interpolates, momentum is Muon's separate EMA.

    Decay acts on the base z. The first averaging coefficient is 1/2,
    unlike ScheduleFreeAdamW's squared-rate weights. Optional linear warmup
    scales the native rate; a constant rate afterwards needs no scheduler.
    Call eval() for averaged validation and train() before the next forward.
    """

    def __init__(  # noqa: PLR0913, PLR0917 - Explicit composed-optimizer hyperparameters.
        self,
        params: Params,
        lr: float = 1e-3,
        beta: float = 0.9,
        momentum: float = 0.95,
        auxiliary: float = 0.05,
        weight_decay: float = 0.0,
        warmup: int = 0,
        *,
        matrix_params: Iterable[Tensor] | None = None,
    ) -> None:
        """Select matrices and initialize query, evaluation and native state."""
        settings(lr, momentum=momentum, auxiliary=auxiliary, weight_decay=weight_decay)
        if type(warmup) is not int or warmup < 0:
            msg = "warmup must be a nonnegative integer"
            raise ValueError(msg)
        model = Parameters(params, matrix_params)
        super().__init__(model.parameters(), lr, beta)
        self.param_groups[0].update(
            updates=0,
            momentum=momentum,
            auxiliary=auxiliary,
            weight_decay=weight_decay,
            warmup=warmup,
        )
        self.spec = Options(self)
        matrices = set(model.hidden_matrices())
        for p in self.param_groups[0]["params"]:
            # The shared outer clock leaves `step` available to auxiliary Adam.
            del self.state[p]["step"]
            self.state[p]["matrix_update"] = p in matrices

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        """Take a native base step using the query gradient, then average it."""
        loss, parameters = self._gradients(closure)
        group = self.param_groups[0]
        group["updates"] += 1
        warmup = min(1.0, group["updates"] / group["warmup"]) if group["warmup"] else 1.0
        eta = group["lr"] * warmup
        coefficient = 1 / (group["updates"] + 1)
        for p in parameters:
            state = self.state[p]
            direction = muon_direction(
                p, state, self.spec, self.workspace, matrix_update=state["matrix_update"]
            )
            state["base"].mul_(1 - eta * group["weight_decay"]).add_(direction, alpha=-eta)
            state["evaluation"].lerp_(state["base"], coefficient)
        self.train()
        return loss
