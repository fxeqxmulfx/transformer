"""Schedule-Free AdamW with the authors' stability revision from version 1.3.

Source: facebookresearch/schedule_free, commit 59ed9b662d5ec5799aded7390b2482758423653c;
the published 1.4.1 adamw_schedulefree.py is identical. Lean correspondence:
Transformer.ScheduleFree.AdamW.trainingStep, trainingRun_pack and evaluation_pack
(e7ee4beb). Default averaging powers are r=0 and weight_lr_power=2.

Local deviations: one group, all trainable parameters need dense real gradients,
eager state initialization, training mode initially, lr=1e-3, unchanged gradients,
and one reusable workspace instead of allocating foreach denominators. Modules
and tensor iterables are accepted. The recurrence uses ordinary scalar kernels.
"""

from __future__ import annotations

import torch

from ._parameters import Parameters, settings
from ._types import Closure, Loss, Params, StateDict  # noqa: TC001 - Runtime annotation introspection.
from ._workspace import BufferedOptimizer


class ScheduleFreeAdamW(BufferedOptimizer):
    """Bias-corrected variance, query decay and squared maximum-rate averaging.

    With rate=lr*warmup(t), m=max(previous_m,rate), c=m²/sum(m²), and
    u=g/(sqrt(v/(1-beta2**t))+eps)+weight_decay*y, the updates are
    y += c*(z-y)+rate*(beta1*(1-c)-1)*u and z -= rate*u, using old y,z.
    Only z and raw v are retained; parameters hold y during training.

    eval() recovers x by y += (1-1/beta1)*(z-y); train() restores y by
    x += (1-beta1)*(z-x). Repeated calls are idempotent, but mode round
    trips can round. Save model and optimizer in the same mode. A zero
    current rate freezes z while x keeps averaging after a positive rate.
    Use ScheduleFreeAdamWPaper for checkpoints of the previous paper variant.
    """

    def __init__(  # noqa: PLR0913, PLR0917 - Preserve the public optimizer signature.
        self,
        params: Params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 0.0,
        eps: float = 1e-8,
        warmup: int = 0,
    ) -> None:
        """Initialize z at the weights, raw v at zero and the shared step clock."""
        settings(lr, betas=betas, weight_decay=weight_decay, eps=eps)
        if betas[0] == 0:
            msg = "beta1 must lie in (0, 1): eval() divides by beta1"
            raise ValueError(msg)
        if type(warmup) is not int or warmup < 0:
            msg = "warmup must be a nonnegative integer"
            raise ValueError(msg)
        parameters = list(Parameters(params).parameters())
        if any(p.layout != torch.strided or p.is_complex() for p in parameters):
            msg = "ScheduleFreeAdamW requires dense real parameters"
            raise ValueError(msg)
        super().__init__(
            parameters,
            {
                "lr": lr,
                "betas": betas,
                "weight_decay": weight_decay,
                "eps": eps,
                "warmup": warmup,
                "training": True,
                "step": 0,
                "maximum": 0.0,
                "total": 0.0,
            },
        )
        for p in parameters:
            self.state[p] = {"base": p.detach().clone(), "second": torch.zeros_like(p)}

    def add_param_group(self, param_group: StateDict) -> None:
        """Register the single supported parameter group."""
        if self.param_groups:
            msg = "ScheduleFreeAdamW supports one parameter group"
            raise ValueError(msg)
        super().add_param_group(param_group)

    def load_state_dict(self, state_dict: StateDict) -> None:
        """Restore revised histories; paper checkpoints need ScheduleFreeAdamWPaper."""
        if any("maximum" not in group for group in state_dict["param_groups"]):
            msg = "Use ScheduleFreeAdamWPaper to resume a paper-variant checkpoint"
            raise ValueError(msg)
        super().load_state_dict(state_dict)

    @torch.no_grad()
    def _mode(self, *, training: bool) -> ScheduleFreeAdamW:
        group = self.param_groups[0]
        if group["training"] != training:
            beta1 = group["betas"][0]
            for p in group["params"]:
                p.lerp_(self.state[p]["base"], 1 - beta1 if training else 1 - 1 / beta1)
            group["training"] = training
        return self

    def train(self) -> ScheduleFreeAdamW:
        """Restore the training query y from the evaluation weights x."""
        return self._mode(training=True)

    def eval(self) -> ScheduleFreeAdamW:
        """Recover the evaluation average x from the training query y."""
        return self._mode(training=False)

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        """Update raw v, y and z from every gradient at the current query y."""
        group = self.param_groups[0]
        if not group["training"]:
            msg = "Call optimizer.train() before a training forward/step"
            raise RuntimeError(msg)
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        parameters = group["params"]
        if any(p.grad is None or p.grad.layout != torch.strided for p in parameters):
            msg = "ScheduleFreeAdamW requires a dense gradient for every parameter"
            raise ValueError(msg)
        beta1, beta2 = group["betas"]
        group["step"] += 1
        step = group["step"]
        ramp = min(1.0, step / group["warmup"]) if group["warmup"] else 1.0
        rate = group["lr"] * ramp
        group["maximum"] = max(rate, group["maximum"])
        group["total"] += group["maximum"] ** 2
        coefficient = group["maximum"] ** 2 / group["total"] if group["total"] else 0.0
        for p in parameters:
            state = self.state[p]
            state["second"].mul_(beta2).addcmul_(p.grad, p.grad, value=1 - beta2)
            work = self.workspace.like(p, "work")
            torch.div(state["second"], 1 - beta2**step, out=work)
            work.sqrt_().add_(group["eps"])
            torch.div(p.grad, work, out=work)
            work.add_(p, alpha=group["weight_decay"])
            p.lerp_(state["base"], coefficient)
            p.add_(work, alpha=rate * (beta1 * (1 - coefficient) - 1))
            state["base"].sub_(work, alpha=rate)
        return loss
