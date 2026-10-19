"""Small independent parameter layouts for warmup and graph-continuation checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

import lean_optim as opt

if TYPE_CHECKING:
    from collections.abc import Callable


def factories() -> dict[str, Callable[..., torch.optim.Optimizer]]:
    """Exercise independent, guarded and explicitly overridden MD direction rates."""
    return {
        "MD": opt.AMSGradMD,
        "guard": lambda p, **kw: opt.AMSGradMD(p, guard_sigma=0.5, **kw),
        "override": lambda p, **kw: opt.AMSGradMD(p, guard_sigma=0.5, direction_rate=0.0625, **kw),
        "SFMD": opt.ScheduleFreeAMSGradMD,
    }


def parameters(device: str = "cpu", dtype: torch.dtype = torch.float64) -> list[torch.Tensor]:
    """Use nonzero matrices and auxiliary tensors with exactly representable entries."""
    return [
        torch.nn.Parameter(
            torch.arange(1, 1 + torch.Size(shape).numel(), dtype=dtype, device=device)
            .reshape(shape)
            .div_(8)
        )
        for shape in ((2, 3), (2,))
    ]


def gradients(optimizer: torch.optim.Optimizer, value: float) -> None:
    """Replace gradients outside capture and allocation measurement."""
    for parameter in optimizer.param_groups[0]["params"]:
        parameter.grad = torch.full_like(parameter, value)


@dataclass
class Run:
    """One optimizer, its parameters, its live device rate and its host scheduler."""

    parameters: list[torch.Tensor]
    optimizer: torch.optim.Optimizer
    scheduler: opt.LearningRateSchedule
    rate: torch.Tensor

    @classmethod
    def create(cls, name: str, device: str, dtype: torch.dtype = torch.float64) -> Run:
        """Construct matching capturable and eager cases on CPU or CUDA."""
        values = parameters(device, dtype)
        rate = torch.tensor(0.125, device=device)
        if name == "AdamX":
            optimizer = opt.AdamX(values, rate=rate)
        elif name == "native":
            optimizer = opt.AdamW(values, lr=rate, capturable=device == "cuda")
        else:
            optimizer = factories()[name](
                values, lr=0.125, gain_rate=0.03125, auxiliary_rate=0.015625, rate=rate
            )
        scheduler = opt.LearningRateSchedule(
            optimizer, warmup=8, total_steps=20, decay="cosine", final_factor=0.125
        )
        return cls(values, optimizer, scheduler, rate)

    def step(self, value: float = 0.125) -> None:
        """Advance gradients, optimizer and schedule in the documented order."""
        gradients(self.optimizer, value)
        self.optimizer.step()
        self.scheduler.step()
