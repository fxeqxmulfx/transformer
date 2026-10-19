"""Couple MD component rates to an attached PyTorch learning-rate scheduler.

The paper callbacks accept independent rates. A scheduler explicitly adds
the common multiplier lr / initial_lr; without one those rates stay independent.
Tensor multipliers use persistent scalar buffers so graph replay reads live rates.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from ._types import Scalar
    from ._workspace import BufferedOptimizer, Workspace


def validate_rate(rate: Scalar) -> None:
    """Validate one real, finite, nonnegative rate before creating optimizer state.

    Read a Tensor once during construction, preserving its caller-owned
    storage. Per-step updates and CUDA graph replays do not call this check.
    """
    if isinstance(rate, Tensor):
        if rate.numel() != 1 or rate.is_complex():
            msg = "Learning rate tensor must contain one real value"
            raise ValueError(msg)
        rate = rate.item()
    if not math.isfinite(rate) or rate < 0:
        msg = "Learning rate must be finite and nonnegative"
        raise ValueError(msg)


def copy_scalar(output: Tensor, value: Scalar) -> Tensor:
    """Write a scalar without replacing its storage."""
    return output.copy_(value.reshape(())) if isinstance(value, Tensor) else output.fill_(value)


def _factor(optimizer: BufferedOptimizer) -> Scalar:
    group = optimizer.param_groups[0]
    current, initial = group["lr"], group["initial_lr"]
    if not isinstance(current, Tensor) and not isinstance(initial, Tensor):
        return current / initial if initial else float(current != 0)
    workspace, reference = optimizer.workspace, group["params"][0]
    current_value = copy_scalar(
        workspace.tensor(reference, "rates.current", (), torch.float64), current
    )
    initial_value = copy_scalar(
        workspace.tensor(reference, "rates.initial", (), torch.float64), initial
    )
    positive = workspace.tensor(reference, "rates.positive", (), torch.bool)
    nonzero = workspace.tensor(reference, "rates.nonzero", (), torch.bool)
    torch.gt(initial_value, 0, out=positive)
    torch.ne(current_value, 0, out=nonzero)
    one = workspace.tensor(reference, "rates.one", (), torch.float64).fill_(1)
    torch.where(positive, initial_value, one, out=initial_value)
    factor = torch.div(current_value, initial_value, out=current_value)
    fallback = workspace.tensor(reference, "rates.fallback", (), torch.float64).copy_(nonzero)
    return torch.where(positive, factor, fallback, out=factor)


def component_rates(optimizer: BufferedOptimizer, *names: str) -> tuple[Scalar | None, ...]:
    """Return independently configured rates times the scheduler's common factor."""
    group = optimizer.param_groups[0]
    values = tuple(
        group[name] if name in group else getattr(optimizer.spec, name) for name in names
    )
    if "initial_lr" not in group:
        return values
    factor = _factor(optimizer)
    result = []
    for name, value in zip(names, values, strict=True):
        if value is None:
            result.append(None)
        elif isinstance(factor, Tensor) or isinstance(value, Tensor):
            output = optimizer.workspace.tensor(
                group["params"][0], f"rates.{name}", (), torch.float64
            )
            result.append(copy_scalar(output, value).mul_(factor))
        else:
            result.append(value * factor)
    return tuple(result)


def active_rates(
    workspace: Workspace, reference: Tensor, name: str, *rates: Scalar
) -> bool | Tensor:
    """Test whether an iterate can move, without synchronizing device scalars."""
    if any(not isinstance(rate, Tensor) and rate != 0 for rate in rates):
        return True
    tensors = [rate for rate in rates if isinstance(rate, Tensor)]
    if not tensors:
        return False
    active = workspace.tensor(reference, name, (), torch.bool).zero_()
    work = workspace.tensor(reference, f"{name}.work", (), torch.bool)
    for rate in tensors:
        torch.ne(rate.reshape(()), 0, out=work)
        torch.logical_or(active, work, out=active)
    return active


def copy_active(output: Tensor, value: Tensor, *, active: bool | Tensor) -> None:
    """Preserve an exactly frozen iterate, including its original floating-point bits."""
    if isinstance(active, Tensor):
        torch.where(active, value, output, out=output)
    elif active:
        output.copy_(value)
