"""Schedule-Free methods, arXiv:2405.15682v4, Sections 2 and 4.4.

SGD implements mix/query/average of Transformer.ScheduleFree.Section2_Averaging,
including the initial base point in the equal-weighted average. AdamW follows
Transformer.ScheduleFree.adamStep exactly, including epsilon placement, decay
at the query, warmup and squared effective-rate weights. These explicit-buffer
implementations store the query in parameters during training and evaluation
weights after eval(). They do not use the upstream package's algebraic buffer
compression or its extra weighting options.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

from ._parameters import Parameters, settings
from ._workspace import BufferedOptimizer

if TYPE_CHECKING:
    from ._types import Closure, Loss, Params, StateDict


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
        beta = group.get("beta", group.get("betas", (0.9,))[0])
        for p in group["params"]:
            state = self.state[p]
            if training:
                work = self.workspace.like(p, "work")
                torch.mul(state["evaluation"], beta, out=p)
                torch.mul(state["base"], 1 - beta, out=work)
                p.add_(work)
            else:
                p.copy_(state["evaluation"])
        return self

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
            s["evaluation"].mul_(1 - c).add_(s["base"], alpha=c)
        self.train()
        return loss


class ScheduleFreeAdamW(ScheduleFreeSGD):
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
            self.state[p].update(second=torch.zeros_like(p), total=0.0)

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        """Apply one update, evaluating an optional closure with gradients enabled."""
        loss, parameters = self._gradients(closure)
        group = self.param_groups[0]
        beta2 = group["betas"][1]
        for p in parameters:
            s, g = self.state[p], p.grad
            s["step"] += 1
            warmup = min(1.0, s["step"] / group["warmup"]) if group["warmup"] else 1.0
            eta = group["lr"] * (1 - beta2 ** s["step"]) ** 0.5 * warmup
            s["second"].mul_(beta2).addcmul_(g, g, value=1 - beta2)
            s["total"] += eta**2
            c = eta**2 / s["total"] if s["total"] else 0.0
            # p is the old query y, needed for the source's independent decay.
            direction = self.workspace.like(p, "direction")
            work = self.workspace.like(p, "work")
            torch.sqrt(s["second"], out=work)
            work.add_(group["eps"])
            torch.mul(g, eta, out=direction)
            direction.div_(work)
            torch.mul(p, eta * group["weight_decay"], out=work)
            direction.add_(work)
            s["base"].sub_(direction)
            s["evaluation"].mul_(1 - c).add_(s["base"], alpha=c)
        self.train()
        return loss
