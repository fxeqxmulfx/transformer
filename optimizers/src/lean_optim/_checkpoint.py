"""Restore live rate storage and the custom kernels' float64 update clocks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

from ._rates import copy_scalar

if TYPE_CHECKING:
    from torch.optim import Optimizer

    from ._types import Scalar


def rate_references(optimizer: Optimizer) -> list[dict[str, Scalar]]:
    """Remember rate tensors whose addresses can be held by callers and CUDA graphs."""
    return [
        {
            name: group[name]
            for name in ("lr", "initial_lr")
            if isinstance(group.get(name), Tensor) or (name == "initial_lr" and name in group)
        }
        for group in optimizer.param_groups
    ]


def restore_rates(optimizer: Optimizer, references: list[dict[str, Scalar]]) -> None:
    """Copy restored rates into destination storage and move new scalar metadata."""
    for group, targets in zip(optimizer.param_groups, references, strict=True):
        for name in ("lr", "initial_lr"):
            value = group.get(name)
            target = targets.get(name)
            if isinstance(target, Tensor):
                if value is not None:
                    copy_scalar(target, value)
                group[name] = target
            elif value is None and target is not None:
                group[name] = target
            elif isinstance(value, Tensor) and group["params"]:
                group[name] = value.to(device=group["params"][0].device)
    if hasattr(optimizer, "capturable"):
        optimizer.capturable = isinstance(optimizer.param_groups[0]["lr"], Tensor)


def restore_clocks(optimizer: Optimizer) -> None:
    """Keep custom clocks on the parameter device without rounding them to float32."""
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            state = optimizer.state.get(parameter, {})
            step = state.get("step")
            if isinstance(step, Tensor):
                state["step"] = step.to(device=parameter.device, dtype=torch.float64)
