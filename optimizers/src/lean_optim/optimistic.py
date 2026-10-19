"""Appendix D.1 clipped optimistic method, arXiv:2405.15682v4.

This is Transformer.ScheduleFree.optimisticStep, using both ball projections
and the old peak for clipping. The stored state starts at zero, as in
optimisticRun. The ball is joint over every supplied tensor, rather than
one ball per tensor. Zero real division is reproduced with safe denominators.
"""

import math

import torch

from ._parameters import Parameters


def project_ball(vector, radius):
    norm = vector.norm()
    return vector * torch.minimum(torch.ones_like(norm), radius / torch.where(norm > 0, norm, 1.0))


class ClippedOptimistic(torch.optim.Optimizer):
    """Transformer.ScheduleFree.optimisticRun on a joint radius-D ball.

    All supplied parameters must start at zero, on one device and dtype.
    This method chooses its own learning rate. It is distinct from generic
    gradient norm clipping and does not support an external lr scheduler.
    """
    def __init__(self, params, radius=1.0):
        if not math.isfinite(radius) or radius < 0:
            raise ValueError("radius must be finite and nonnegative")
        model = Parameters(params)
        parameters = list(model.parameters())
        if any(p.device != parameters[0].device or p.dtype != parameters[0].dtype for p in parameters):
            raise ValueError("The joint optimistic ball needs one device and dtype")
        if any(bool((p != 0).any()) for p in parameters):
            raise ValueError("optimisticRun starts at zero; initialize all parameters to zero")
        super().__init__(parameters, dict(radius=radius))
        packed = torch.cat([p.detach().flatten() for p in parameters])
        self.state["optimistic"] = dict(auxiliary=torch.zeros_like(packed), previous=torch.zeros_like(packed),
                                        peak=packed.new_zeros(()), energy=packed.new_zeros(()))

    def add_param_group(self, group):
        if self.param_groups:
            raise ValueError("ClippedOptimistic supports one joint parameter group")
        super().add_param_group(group)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        group, s = self.param_groups[0], self.state["optimistic"]
        parameters = group["params"]
        if any(p.grad is None or p.grad.is_sparse for p in parameters):
            raise ValueError("ClippedOptimistic needs a dense gradient for every parameter")
        g = torch.cat([p.grad.flatten() for p in parameters])
        difference = g - s["previous"]
        norm = difference.norm()
        clipped = s["previous"] + torch.minimum(s["peak"], norm) / torch.where(norm > 0, norm, 1.0) * difference
        s["peak"].copy_(torch.maximum(s["peak"], norm))
        s["energy"].add_((clipped - s["previous"]).square().sum())
        scale = (s["peak"].square() + s["energy"]).sqrt()
        eta = torch.where(scale > 0, group["radius"] / torch.where(scale > 0, scale, 1.0), 0.0)
        s["auxiliary"].copy_(project_ball(s["auxiliary"] - eta * clipped, group["radius"]))
        query = project_ball(s["auxiliary"] - eta * g, group["radius"])
        s["previous"].copy_(g)
        offset = 0
        for p in parameters:
            p.copy_(query[offset:offset + p.numel()].view_as(p))
            offset += p.numel()
        return loss
