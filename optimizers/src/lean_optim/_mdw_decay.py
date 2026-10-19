"""Physical MDW correction and finite softplus storage, Transformer.MDW.Basic.

The correction subtracts rate * decay * old_weight outside every moment
history. Repair preserves that physical result and restores the direction
radius when nonzero. A zero result retains finite gains and a zero direction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

from ._rates import copy_active, scalar_rate
from .magnitude import softplus

if TYPE_CHECKING:
    from ._types import Scalar
    from ._workspace import Workspace


class PhysicalDecay:
    """Reuse physical-correction and storage-repair buffers between updates."""

    def __init__(self, workspace: Workspace, *, smooth: bool = False) -> None:
        """Choose the gain convention of the corresponding MD implementation."""
        self.workspace, self.smooth = workspace, smooth

    def _gain(self, point: Tensor) -> Tensor:
        if not self.smooth:
            return softplus(point, self.workspace)
        zero = self.workspace.tensor(point, "mdw.zero", ()).zero_()
        return torch.logaddexp(point, zero, out=self.workspace.like(point, "mdw.gain"))

    def apply(self, weight: Tensor, old: Tensor, rate: Scalar, decay: float) -> Tensor:
        """Return a borrowed corrected weight without changing either input."""
        output = self.workspace.like(weight, "mdw.correction").copy_(old)
        output.mul_(scalar_rate(rate)).mul_(decay)
        return torch.sub(weight, output, out=output)

    def repair(
        self,
        weight: Tensor,
        gains: tuple[Tensor, Tensor],
        radius: Tensor,
        *,
        direction: Tensor | None = None,
        active: bool | Tensor = True,
    ) -> None:
        """Repair row gains and optionally write the explicit projected base direction."""
        if isinstance(active, bool) and not active:
            return
        row, column = gains
        workspace = self.workspace
        recovered = workspace.like(weight, "mdw.direction")
        torch.mul(self._gain(row)[:, None], self._gain(column)[None, :], out=recovered)
        torch.div(weight, recovered, out=recovered)
        norm = workspace.norm(recovered)
        positive = workspace.tensor(norm, "mdw.positive", dtype=torch.bool)
        torch.gt(norm, 0, out=positive)
        scale = torch.div(norm, radius, out=workspace.like(norm, "mdw.scale"))
        gain = torch.mul(self._gain(row), scale, out=workspace.like(row, "mdw.scaled_gain"))
        work = torch.neg(gain, out=workspace.like(row, "mdw.inverse_work"))
        work.expm1_().neg_().log_()
        gain.add_(work)
        if direction is not None:
            torch.div(radius, workspace.safe(norm), out=scale)
            recovered.mul_(scale)
            copy_active(direction, recovered, active=active)
        if isinstance(active, Tensor):
            torch.logical_and(positive, active, out=positive)
        torch.where(positive, gain, row, out=row)
