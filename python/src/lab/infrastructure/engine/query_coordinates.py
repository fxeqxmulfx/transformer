"""
Condition depth-query coordinates without discarding Schedule-Free history.

Source: ScheduleFreeAdamW at 4e58930 and the companion Lean
Transformer.AttentionResiduals.QueryCoordinates. For a positive factor c,
y and z divide by c, the gradient second moment multiplies by c**2, and
the query coordinate scale multiplies by c. The implied average x divides
by c too; beta, averaging weights, step and sampler are unchanged.

This is an explicit change to optimization geometry. Subsequent ordinary
Schedule-Free AdamW steps operate in the new coordinates; their physical
learning rate is therefore different from the unconditioned baseline.
It is not a claim of identical optimizer trajectories or convergence.
"""

from typing import TYPE_CHECKING

import torch


if TYPE_CHECKING:
    from torch import Tensor



def stable_rms(value: Tensor) -> Tensor:
    """Measure RMS without squaring a potentially huge original coordinate."""
    scale = value.abs().amax().clamp_min(torch.finfo(value.dtype).tiny)
    return scale * (value / scale).square().mean().sqrt()


@torch.no_grad()
def recondition_state(
    query: Tensor,
    base: Tensor,
    second: Tensor,
    coordinate_scale: Tensor,
    beta: float,
    maximum_rms: float,
) -> Tensor:
    """Apply one common linear coordinate change to the complete query state."""
    average = torch.lerp(query, base, 1 - 1 / beta)
    largest = torch.stack([stable_rms(v) for v in (query, base, average)]).amax()
    factor = torch.where(largest > maximum_rms, largest, torch.ones_like(largest))
    query.div_(factor)
    base.div_(factor)
    second.mul_(factor).mul_(factor)
    coordinate_scale.mul_(factor)
    return factor
