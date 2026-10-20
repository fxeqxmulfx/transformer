"""Schedule-Free methods, arXiv:2405.15682v4, Sections 2 and 4.4.

SGD implements mix/query/average of Transformer.ScheduleFree.Section2_Averaging,
including the initial base point in the equal-weighted average. AdamWPaper follows
Transformer.ScheduleFree.adamStep exactly, including epsilon placement, decay
at the query, warmup and squared effective-rate weights. These explicit-buffer
implementations store the query in parameters during training and evaluation
weights after eval(). They do not use the upstream package's algebraic buffer
compression or its extra weighting options. Interpolation uses stable lerp
operations; AdamWPaper reuses one denominator scratch tensor per parameter.
The public revised ScheduleFreeAdamW is re-exported from schedule_free_adamw.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

from ._parameters import Parameters, settings
from ._workspace import BufferedOptimizer
from .schedule_free_adamw import ScheduleFreeAdamW

if TYPE_CHECKING:
    from ._types import Closure, Loss, Params, StateDict

__all__ = ["ScheduleFreeAMSGradW", "ScheduleFreeAdamW", "ScheduleFreeAdamWPaper", "ScheduleFreeSGD"]


class ScheduleFreeSGD(BufferedOptimizer):
    """Equal-weighted Schedule-Free SGD; beta may include both 0 and 1.

    Call optimizer.eval() for validation and optimizer.train() before the
    next training forward. The initial point has weight one.
    """

    def __init__(self, params: Params, lr: float = 1e-3, beta: float = 0.9) -> None:
        """Initialize the parameters, hyperparameters and persistent state."""
        if not 0 <= beta <= 1:
            msg = "beta must lie in [0, 1]"
            raise ValueError(msg)
        settings(lr)
        model = Parameters(params)
        super().__init__(model.parameters(), {"lr": lr, "beta": beta, "training": True})
        for p in self.param_groups[0]["params"]:
            self.state[p] = {
                "base": p.detach().clone(),
                "evaluation": p.detach().clone(),
                "step": 0,
            }

    def add_param_group(self, group: StateDict) -> None:
        """Register the supported parameter group."""
        if self.param_groups:
            msg = "Schedule-Free methods support one parameter group"
            raise ValueError(msg)
        super().add_param_group(group)

    @torch.no_grad()
    def _mode(self, *, training: bool) -> ScheduleFreeSGD:
        group = self.param_groups[0]
        group["training"] = training
        for p in group["params"]:
            self._write(p, self.state[p], training=training)
        return self

    def _write(self, output: Tensor, state: StateDict, *, training: bool) -> None:
        group = self.param_groups[0]
        beta = group.get("beta", group.get("betas", (0.9,))[0])
        if training:
            if beta == 0:
                output.copy_(state["base"])
            elif beta == 1:
                output.copy_(state["evaluation"])
            else:
                torch.lerp(state["base"], state["evaluation"], beta, out=output)
        else:
            output.copy_(state["evaluation"])

    def train(self) -> ScheduleFreeSGD:
        """Write the training query into the parameters."""
        return self._mode(training=True)

    def eval(self) -> ScheduleFreeSGD:
        """Write the evaluation average into the parameters."""
        return self._mode(training=False)

    def _gradients(self, closure: Closure) -> tuple[Loss, list[Tensor]]:
        if not self.param_groups[0]["training"]:
            msg = "Call optimizer.train() before a training forward/step"
            raise RuntimeError(msg)
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        parameters = self.param_groups[0]["params"]
        if any(p.grad is None or p.grad.is_sparse for p in parameters):
            msg = "Schedule-Free methods require a dense gradient for every parameter"
            raise ValueError(msg)
        return loss, parameters

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        """Apply one update, evaluating an optional closure with gradients enabled."""
        loss, parameters = self._gradients(closure)
        group = self.param_groups[0]
        for p in parameters:
            s = self.state[p]
            s["base"].add_(p.grad, alpha=-group["lr"])
            s["step"] += 1
            c = 1 / (s["step"] + 1)
            s["evaluation"].lerp_(s["base"], c)
        self.train()
        return loss


class ScheduleFreeAdamWPaper(ScheduleFreeSGD):
    """Section 4.4 Algorithm 1; Transformer.ScheduleFree.adamStep.

    eta_t = lr * sqrt(1-beta2**t) * min(1,t/warmup); epsilon is added to
    sqrt(raw_second), with decay evaluated at y. This is deliberately not
    the epsilon placement of torch.optim.AdamW. No nonconvex convergence
    theorem is claimed by the source or this package.
    """

    def __init__(  # noqa: PLR0913, PLR0917 - Preserve the established optimizer/recurrence signature.
        self,
        params: Params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 0.0,
        eps: float = 1e-8,
        warmup: int = 0,
    ) -> None:
        """Initialize the parameters, hyperparameters and persistent state."""
        settings(lr, betas=betas, weight_decay=weight_decay, eps=eps)
        if type(warmup) is not int or warmup < 0:
            msg = "warmup must be a nonnegative integer"
            raise ValueError(msg)
        super().__init__(params, lr, betas[0])
        self.param_groups[0].update(betas=betas, weight_decay=weight_decay, eps=eps, warmup=warmup)
        del self.param_groups[0]["beta"]
        for p in self.param_groups[0]["params"]:
            self.state[p].update(self._second_state(p))

    def _second_state(self, value: Tensor) -> StateDict:
        return {"second": torch.zeros_like(value), "total": 0.0}

    def _variance(self, state: StateDict, gradient: Tensor) -> Tensor:
        beta2 = self.param_groups[0]["betas"][1]
        state["second"].mul_(beta2).addcmul_(gradient, gradient, value=1 - beta2)
        return state["second"]

    def _advance(
        self,
        query: Tensor,
        gradient: Tensor,
        state: StateDict,
        rate: float,
        *,
        average: bool = True,
    ) -> tuple[float, float] | None:
        group = self.param_groups[0]
        state["step"] += 1
        warmup = min(1.0, state["step"] / group["warmup"]) if group["warmup"] else 1.0
        eta = rate * (1 - group["betas"][1] ** state["step"]) ** 0.5 * warmup
        variance = self._variance(state, gradient)
        state["total"] += eta**2
        c = eta**2 / state["total"] if state["total"] else 0.0
        work = self.workspace.like(query, "work")
        torch.sqrt(variance, out=work)
        work.add_(group["eps"])
        state["base"].addcdiv_(gradient, work, value=-eta)
        state["base"].add_(query, alpha=-eta * state.get("weight_decay", group["weight_decay"]))
        if eta and "radius" in state:
            norm = self.workspace.norm(state["base"])
            scale = torch.div(
                state["radius"],
                self.workspace.safe(norm),
                out=self.workspace.like(norm, "projection.scale"),
            )
            state["base"].mul_(scale)
        if average:
            self._average(state, c)
            return None
        return eta, c

    @staticmethod
    def _average(state: StateDict, coefficient: float) -> None:
        """Average a completed base update after any physical factor-space correction."""
        if coefficient == 1:
            state["evaluation"].copy_(state["base"])
        elif coefficient:
            state["evaluation"].lerp_(state["base"], coefficient)

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        """Apply one update, evaluating an optional closure with gradients enabled."""
        loss, parameters = self._gradients(closure)
        for p in parameters:
            self._advance(p, p.grad, self.state[p], self.param_groups[0]["lr"])
        self.train()
        return loss


class ScheduleFreeAMSGradW(ScheduleFreeAdamWPaper):
    """Schedule-Free AdamW with a running maximum of the raw second moment.

    This package's extension combines arXiv:2405.15682v4, Section 4.4,
    Algorithm 1 with arXiv:1904.03590v4, Algorithm 1 and Section 6.
    beta1 controls interpolation; there is no first-moment EMA. Effective
    rates, epsilon, query decay and squared-rate averaging follow
    ScheduleFreeAdamWPaper. The maximum is taken before bias correction.
    Lean: Transformer.ScheduleFree.AMSGrad.step.
    """

    _use_maximum = True

    def _second_state(self, value: Tensor) -> StateDict:
        state = super()._second_state(value)
        if self._use_maximum:
            state["maximum"] = torch.zeros_like(value)
        return state

    def _variance(self, state: StateDict, gradient: Tensor) -> Tensor:
        variance = super()._variance(state, gradient)
        if not self._use_maximum:
            return variance
        return torch.maximum(state["maximum"], variance, out=state["maximum"])
