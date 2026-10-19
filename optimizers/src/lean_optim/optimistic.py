"""Appendix D.1 clipped optimistic method, arXiv:2405.15682v4.

This is Transformer.ScheduleFree.optimisticStep, using both ball projections
and the old peak for clipping. The stored state starts at zero, as in
optimisticRun. The ball is joint over every supplied tensor, rather than
one ball per tensor. Zero real division is reproduced with safe denominators.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from ._parameters import Parameters
from ._workspace import BufferedOptimizer, Workspace

if TYPE_CHECKING:
    from ._types import Closure, Loss, Params, StateDict


def project_ball(vector: Tensor, radius: float) -> Tensor:
    """Return the Euclidean ball projection, with zero division defined as zero."""
    norm = vector.norm()
    return vector * torch.minimum(torch.ones_like(norm), radius / torch.where(norm > 0, norm, 1.0))


class ClippedOptimistic(BufferedOptimizer):
    """Transformer.ScheduleFree.optimisticRun on a joint radius-D ball.

    All supplied parameters must start at zero, on one device and dtype.
    This method chooses its own learning rate. It is distinct from generic
    gradient norm clipping and does not support an external lr scheduler.
    """

    def __init__(self, params: Params, radius: float = 1.0) -> None:
        """Initialize the parameters, hyperparameters and persistent state."""
        if not math.isfinite(radius) or radius < 0:
            msg = "radius must be finite and nonnegative"
            raise ValueError(msg)
        model = Parameters(params)
        parameters = list(model.parameters())
        if any(
            p.device != parameters[0].device or p.dtype != parameters[0].dtype for p in parameters
        ):
            msg = "The joint optimistic ball needs one device and dtype"
            raise ValueError(msg)
        if any(bool((p != 0).any()) for p in parameters):
            msg = "optimisticRun starts at zero; initialize all parameters to zero"
            raise ValueError(msg)
        super().__init__(parameters, {"radius": radius})
        packed = torch.cat([p.detach().flatten() for p in parameters])
        self.state["optimistic"] = {
            "auxiliary": torch.zeros_like(packed),
            "previous": torch.zeros_like(packed),
            "peak": packed.new_zeros(()),
            "energy": packed.new_zeros(()),
        }

    def add_param_group(self, group: StateDict) -> None:
        """Register the supported parameter group."""
        if self.param_groups:
            msg = "ClippedOptimistic supports one joint parameter group"
            raise ValueError(msg)
        super().add_param_group(group)

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        """Apply one update, evaluating an optional closure with gradients enabled."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        group, s = self.param_groups[0], self.state["optimistic"]
        parameters = group["params"]
        if any(p.grad is None or p.grad.is_sparse for p in parameters):
            msg = "ClippedOptimistic needs a dense gradient for every parameter"
            raise ValueError(msg)
        g = self.workspace.like(s["previous"], "gradient")
        offset = 0
        for p in parameters:
            g[offset : offset + p.numel()].view_as(p).copy_(p.grad)
            offset += p.numel()
        difference = self.workspace.like(g, "difference")
        clipped = self.workspace.like(g, "clipped")
        torch.sub(g, s["previous"], out=difference)
        norm = self.workspace.norm(difference)
        coefficient = torch.minimum(s["peak"], norm, out=self.workspace.like(norm, "coefficient"))
        coefficient.div_(self.workspace.safe(norm))
        torch.mul(difference, coefficient, out=clipped)
        torch.add(s["previous"], clipped, out=clipped)
        torch.maximum(s["peak"], norm, out=s["peak"])
        torch.sub(clipped, s["previous"], out=difference)
        s["energy"].add_(self.workspace.sum(difference.square_()))
        scale = torch.square(s["peak"], out=self.workspace.like(s["peak"], "scale"))
        scale.add_(s["energy"]).sqrt_()
        positive = self.workspace.tensor(scale, "positive", dtype=torch.bool)
        torch.gt(scale, 0, out=positive)
        eta = torch.reciprocal(self.workspace.safe(scale), out=self.workspace.like(scale, "eta"))
        eta.mul_(group["radius"])
        zero = self.workspace.tensor(eta, "zero", ()).zero_()
        torch.where(positive, eta, zero, out=eta)
        clipped.mul_(eta)
        s["auxiliary"].sub_(clipped)
        project_ball_(s["auxiliary"], group["radius"], self.workspace)
        torch.mul(g, eta, out=difference)
        query = torch.sub(s["auxiliary"], difference, out=difference)
        project_ball_(query, group["radius"], self.workspace)
        s["previous"].copy_(g)
        offset = 0
        for p in parameters:
            p.copy_(query[offset : offset + p.numel()].view_as(p))
            offset += p.numel()
        return loss


def project_ball_(vector: Tensor, radius: float, workspace: Workspace) -> None:
    """Project a vector in place, with zero real division reproduced safely."""
    norm = workspace.norm(vector)
    coefficient = torch.reciprocal(workspace.safe(norm), out=workspace.like(norm, "projection"))
    coefficient.mul_(radius)
    one = workspace.tensor(norm, "one", ()).fill_(1)
    torch.minimum(one, coefficient, out=coefficient)
    vector.mul_(coefficient)
