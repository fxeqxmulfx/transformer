"""Small MDW runs shared by continuation and CUDA allocation checks."""

from __future__ import annotations

from dataclasses import dataclass

import torch

import lean_optim as opt

NAMES = ("AMSGradMDW", "AdamMDW", "ScheduleFreeAMSGradMDW", "ScheduleFreeAdamMDW")


@dataclass
class Run:
    """One optimizer and its documented host-side schedule."""

    values: list[torch.Tensor]
    optimizer: torch.optim.Optimizer
    scheduler: opt.LearningRateSchedule
    rate: torch.Tensor | None

    @classmethod
    def create(
        cls,
        name: str,
        device: str = "cpu",
        dtype: torch.dtype = torch.float64,
        shape: tuple[int, int] = (6, 8),
    ) -> Run:
        """Construct matching nonzero matrix/vector layouts with distinct component rates."""
        values = [
            torch.nn.Parameter(
                torch.arange(1, shape[0] * shape[1] + 1, device=device, dtype=dtype).reshape(shape)
                / 64
            ),
            torch.nn.Parameter(torch.arange(1, shape[0] + 1, device=device, dtype=dtype) / 16),
        ]
        rate = None if name.startswith("ScheduleFree") else torch.tensor(0.125, device=device)
        optimizer = getattr(opt, name)(
            values,
            lr=0.125,
            gain_rate=0.03125,
            auxiliary_rate=0.015625,
            betas=(0.5, 0.75),
            eps=0.125,
            weight_decay=0.25,
            **({} if rate is None else {"rate": rate}),
        )
        scheduler = opt.LearningRateSchedule(
            optimizer, warmup=8, total_steps=20, decay="cosine", final_factor=0.125
        )
        run = cls(values, optimizer, scheduler, rate)
        run.gradients()
        return run

    def gradients(self, value: float = 0.125) -> None:
        """Install fixed-shaped gradients outside measured or captured updates."""
        for parameter in self.values:
            if parameter.grad is None:
                parameter.grad = torch.full_like(parameter, value)
            else:
                parameter.grad.fill_(value)

    def step(self) -> None:
        """Advance the optimizer and then the host-side schedule."""
        self.optimizer.step()
        self.scheduler.step()
