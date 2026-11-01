"""Foreach Schedule-Free AdamW for task145, ported from diff_gpt ad76812.

Same revised recurrence and complete state as lean_optim.ScheduleFreeAdamW;
only kernel grouping/allocation differs. Every variant in this new cohort
uses this backend. Historical experiment optimizer paths stay unchanged.
Source: diff_gpt/domain/optimizer/schedule_free.py, arXiv:2405.15682v4 §4.4,
Lean Transformer.ScheduleFree.AdamW.trainingRun_pack/evaluation_pack.
"""

import torch
from lean_optim import ScheduleFreeAdamW


class ResidualScheduleFreeAdamW(ScheduleFreeAdamW):
    @torch.no_grad()
    def _mode(self, *, training):
        group = self.param_groups[0]
        if group["training"] != training:
            beta = group["betas"][0]
            parameters = group["params"]
            torch._foreach_lerp_(parameters, [self.state[p]["base"] for p in parameters],
                                 1 - beta if training else 1 - 1 / beta)
            group["training"] = training
        return self

    @torch.no_grad()
    def step(self, closure=None):
        """
        Update v, z and y from the gradients at y.

        `closure`, if given, recomputes the loss with gradients enabled; its
        loss is returned. RuntimeError in eval mode; ValueError when a
        parameter has no gradient or a sparse one.
        """
        group = self.param_groups[0]
        if not group["training"]:
            msg = "call train() before step(): the parameters hold x, not y"
            raise RuntimeError(msg)
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        params = group["params"]
        grads = [p.grad for p in params]
        if any(g is None or g.is_sparse for g in grads):
            msg = "ScheduleFreeAdamW needs a dense gradient for every parameter"
            raise ValueError(msg)
        beta1, beta2 = group["betas"]
        group["step"] += 1
        t = group["step"]
        ramp = min(1.0, t / group["warmup"]) if group["warmup"] else 1.0
        rate = group["lr"] * ramp
        group["maximum"] = max(rate, group["maximum"])
        group["total"] += group["maximum"] ** 2
        weight = group["maximum"] ** 2 / group["total"] if group["total"] else 0.0
        second = [self.state[p]["second"] for p in params]
        base = [self.state[p]["base"] for p in params]
        torch._foreach_mul_(second, beta2)
        torch._foreach_addcmul_(second, grads, grads, value=1 - beta2)
        # torch exposes the foreach kernels only as private functions.
        denominator = torch._foreach_div(second, 1 - beta2**t)
        torch._foreach_sqrt_(denominator)
        torch._foreach_add_(denominator, group["eps"])
        update = torch._foreach_div(grads, denominator)
        # The decay acts at y, which the parameters hold.
        torch._foreach_add_(update, params, alpha=group["weight_decay"])
        # y follows x and z, from the z before the step.
        torch._foreach_lerp_(params, base, weight)
        torch._foreach_add_(params, update, alpha=rate * (beta1 * (1 - weight) - 1))
        torch._foreach_sub_(base, update, alpha=rate)
        return loss
