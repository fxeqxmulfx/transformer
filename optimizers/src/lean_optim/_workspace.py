"""Reusable, non-checkpointed work tensors for eager and captured updates.

Buffers are fully overwritten before use. They are not mathematical state:
loading a checkpoint discards them, and no checkpoint grows with workspace.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

from ._checkpoint import rate_references, restore_clocks, restore_rates

if TYPE_CHECKING:
    from collections.abc import Iterable

    from ._types import StateDict


class Workspace:
    """Cache named buffers by their reference tensor, outside optimizer state."""

    def __init__(self) -> None:
        """Initialize the parameters, hyperparameters and persistent state."""
        self.buffers = {}

    def like(self, reference: Tensor, name: str) -> Tensor:
        """Return a named work tensor matching the reference shape and layout."""
        return self.tensor(reference, name)

    def tensor(
        self,
        reference: Tensor,
        name: str,
        shape: tuple[int, ...] | torch.Size | None = None,
        dtype: torch.dtype | None = None,
    ) -> Tensor:
        """Return a named work tensor with the specified shape and dtype."""
        group = self.buffers.get(reference)
        if group is None:
            group = self.buffers[reference] = {}
        buffer = group.get(name)
        target_shape = reference.shape if shape is None else shape
        target_dtype = reference.dtype if dtype is None else dtype
        if (
            buffer is None
            or buffer.shape != target_shape
            or buffer.dtype != target_dtype
            or buffer.device != reference.device
        ):
            buffer = group[name] = (
                torch.empty_like(reference)
                if shape is None and dtype is None
                else torch.empty(target_shape, dtype=target_dtype, device=reference.device)
            )
        return buffer

    def identity(self, reference: Tensor) -> Tensor:
        """Return a reusable identity matrix on the reference device and dtype."""
        size = reference.size(-1)
        result = self.tensor(reference, "identity", (size, size))
        # Identity is small relative to a batched history. Filling it also
        # handles an in-place change to the reference's device or dtype.
        result.zero_().diagonal().fill_(1)
        return result

    def safe(self, value: Tensor) -> Tensor:
        """Copy a positive denominator, substituting one at zero."""
        mask = self.tensor(value, "safe.positive", dtype=torch.bool)
        torch.gt(value, 0, out=mask)
        one = self.tensor(value, "safe.one", ()).fill_(1)
        return torch.where(mask, value, one, out=self.like(value, "safe.denominator"))

    def norm(self, value: Tensor, *, reference: Tensor | None = None, name: str = "norm") -> Tensor:
        """Compute a vector norm without allocating its scalar output."""
        return torch.linalg.vector_norm(
            value, out=self.tensor(value if reference is None else reference, name, ())
        )

    def sum(self, value: Tensor, *, reference: Tensor | None = None, name: str = "sum") -> Tensor:
        """Compute a sum without allocating its scalar output."""
        return torch.sum(
            value, dim=None, out=self.tensor(value if reference is None else reference, name, ())
        )


def joint_metrics(
    workspace: Workspace,
    parameters: list[Tensor],
    directions: list[Tensor],
    *,
    work_name: str = "metrics.work",
) -> tuple[Tensor, Tensor, Tensor]:
    """Compute joint squared norms and alignment in parameter order.

    Each prefix retains the dtype of ordinary Python sum at that point,
    including promotion between parameters of different floating dtypes.
    """
    results = []
    for kind, name in (
        ("gradient", "metrics.gradient"),
        ("direction", "metrics.direction"),
        ("alignment", "metrics.alignment"),
    ):
        total = None
        for parameter, direction in zip(parameters, directions, strict=True):
            work = workspace.like(parameter, work_name)
            if kind == "gradient":
                torch.square(parameter.grad, out=work)
            elif kind == "direction":
                torch.square(direction, out=work)
            else:
                torch.mul(parameter.grad, direction, out=work)
            value = workspace.sum(work)
            dtype = value.dtype if total is None else torch.promote_types(total.dtype, value.dtype)
            output = workspace.tensor(parameter, name, (), dtype)
            torch.add(0 if total is None else total, value, out=output)
            total = output
        results.append(total)
    return results[0], results[1], results[2]


def descent_test(
    workspace: Workspace, metrics: tuple[Tensor, Tensor, Tensor], sigma: float
) -> Tensor:
    """Evaluate the descent certificate without allocating boolean scalars."""
    norm_g, norm_d, alignment = metrics
    threshold = torch.mul(norm_g, sigma, out=workspace.like(norm_g, "threshold"))
    accepted = workspace.tensor(alignment, "accepted", dtype=torch.bool)
    length = workspace.tensor(norm_d, "length", dtype=torch.bool)
    torch.ge(alignment, threshold, out=accepted)
    torch.le(norm_d, norm_g, out=length)
    return torch.logical_and(accepted, length, out=accepted)


class BufferedOptimizer(torch.optim.Optimizer):
    """An optimizer whose temporary work tensors are reused between steps."""

    def __init__(self, params: Iterable[Tensor], defaults: StateDict) -> None:
        """Initialize the parameters, hyperparameters and persistent state."""
        super().__init__(params, defaults)
        self.workspace = Workspace()

    def load_state_dict(self, state_dict: StateDict) -> None:
        """Restore shared tensors to their constructed placement and discard scratch."""
        shared = {name: state for name, state in self.state.items() if isinstance(name, str)}
        rates = rate_references(self)
        super().load_state_dict(state_dict)
        # PyTorch casts parameter states, but copies string-keyed state as is.
        # Each constructed tensor supplies its own dtype/device, including
        # int64 counters and MAGMA scales for matrices with different dtypes.
        for name, template in shared.items():
            if name in self.state:
                self.state[name] = {
                    key: value.to(template[key])
                    if isinstance(value, Tensor) and isinstance(template.get(key), Tensor)
                    else value
                    for key, value in self.state[name].items()
                }
        restore_rates(self, rates)
        restore_clocks(self)
        self.workspace.buffers.clear()
