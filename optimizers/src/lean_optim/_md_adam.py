"""Bias-corrected Adam factor callback, Transformer.MDW.adamVectorCallback.

Native Adam's epsilon follows sqrt(bias-corrected variance). A single
float64 clock advances once per parameter, before its three factor updates.
The callback has no decay or maximum buffer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

from .coordinate import bias_correction

if TYPE_CHECKING:
    from ._types import Scalar, TensorState
    from ._workspace import Workspace


def adam(  # noqa: PLR0913, PLR0917 - Match the existing shared MD factor callback interface.
    point: Tensor,
    gradient: Tensor,
    state: TensorState,
    prefix: str,
    rate: Scalar,
    beta: float,
    beta2: float,
    eps: float,
    workspace: Workspace | None = None,
    *,
    output: Tensor | None = None,
    denominator: Tensor | None = None,
) -> Tensor:
    """Update genuine Adam moments, clock and point without a weight penalty."""
    if prefix == "":
        if "step" not in state:
            state["step"] = point.new_zeros((), dtype=torch.float64)
        state["step"].add_(1)
    step = state["step"]
    m, v = state[prefix + "m"], state[prefix + "v"]
    m.mul_(beta).add_(gradient, alpha=1 - beta)
    v.mul_(beta2).addcmul_(gradient, gradient, value=1 - beta2)
    if workspace is None:
        return point - rate * (m / (1 - beta**step)) / ((v / (1 - beta2**step)).sqrt() + eps)
    if output is None:
        output = workspace.like(point, "adam.output")
    if denominator is None:
        denominator = workspace.like(point, "adam.denominator")
    torch.div(m, bias_correction(beta, step, workspace, "mdw.bias1"), out=output)
    torch.div(v, bias_correction(beta2, step, workspace, "mdw.bias2"), out=denominator)
    denominator.sqrt_().add_(eps)
    output.mul_(rate).div_(denominator)
    return torch.sub(point, output, out=output)
