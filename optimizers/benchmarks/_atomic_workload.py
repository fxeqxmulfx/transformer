# Copyright (c) 2026 Mikhail Borisov
"""Small physical order-matching family for profiling AtomicColumns.

This is the two-token final-position case of matchingHeadOutput and
AtomicMatchingContext.lean. The binary sparsemax is evaluated in closed form;
at support boundaries its gradient is zero, as in the lab's sparsemax kernel.
Only the final causal position is evaluated. No lab module is imported.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from types import ModuleType


def _output(
    q: torch.Tensor, k: torch.Tensor, values: torch.Tensor, tokens: torch.Tensor
) -> torch.Tensor:
    query = q[tokens[:, -1]]
    scores = (query[:, None] * k[tokens]).sum(-1)
    linear = (scores[:, 0] - scores[:, 1] + 1) / 2
    first = torch.where(linear <= 0, 0.0, torch.where(linear >= 1, 1.0, linear))
    return first[:, None] * values[tokens[:, 0]] + (1 - first[:, None]) * values[tokens[:, 1]]


class _Head(torch.nn.Module):
    def __init__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        values: torch.Tensor,
        *,
        forward_map: Callable[..., torch.Tensor] = _output,
    ) -> None:
        super().__init__()
        self.q, self.k, self.values = (
            torch.nn.Parameter(t.detach().clone()) for t in (q, k, values)
        )
        self.forward_map = forward_map

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.forward_map(self.q, self.k, self.values, tokens)


class _Model(torch.nn.Module):
    def __init__(self, device: str) -> None:
        super().__init__()
        self.spec = SimpleNamespace(heads=3, cap=1.0)
        self.forward_map = _output
        self.q = torch.nn.Parameter(torch.zeros(3, 2, 1, dtype=torch.float64, device=device))
        self.k = torch.nn.Parameter(torch.zeros_like(self.q))
        self.values = torch.nn.Parameter(torch.zeros_like(self.q))
        self.register_buffer("mass", torch.zeros(3, dtype=torch.float64, device=device))
        self.register_buffer("active", torch.tensor(1, device=device))
        with torch.no_grad():
            self.q[0].fill_(0.25)
            self.k[0].copy_(self.k.new_tensor([[-0.5], [0.5]]))
            self.values[0].fill_(0.75)
            self.mass[0] = 1

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return sum(
            self.mass[i] * _output(self.q[i], self.k[i], self.values[i], tokens)
            for i in range(int(self.active))
        )

    @torch.no_grad()
    def append(self, head: _Head) -> None:
        index = int(self.active)
        for name in ("q", "k", "values"):
            getattr(self, name)[index].copy_(getattr(head, name))
        self.mass[index] = 0
        self.active.add_(1)

    @torch.no_grad()
    def prune(self) -> None:
        count = int(self.active)
        keep = (self.mass[:count] > 0).nonzero().flatten()
        for tensor in (self.q, self.k, self.values, self.mass):
            tensor[: len(keep)].copy_(tensor[:count].index_select(0, keep))
            tensor[len(keep) : count].zero_()
        self.active.fill_(len(keep))


class _Task:
    def __init__(self, device: str) -> None:
        self.tokens = torch.tensor([[0, 1], [1, 0]], device=device)
        self.targets = torch.tensor([[1.0], [-1.0]], dtype=torch.float64, device=device)

    def forward(
        self, model: _Model | _Head, batch: torch.Tensor, *, supervised: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not supervised:
            msg = "This workload profiles supervised physical matching"
            raise ValueError(msg)
        return model(self.tokens[batch]), self.targets[batch]

    @staticmethod
    def loss(output: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return (output - targets).square().mean()


def cases(
    device: str, optim: ModuleType
) -> Iterator[tuple[str, object, list[torch.Tensor], Callable[[], object]]]:
    """Profile exact scalar correction on CPU and a small autograd search on both devices."""
    modes = ("order", "search") if device == "cpu" else ("search",)
    for mode in modes:
        model, task = _Model(device), _Task(device)
        optimizer = optim.AtomicColumns(
            model, task, head_factory=_Head, pricing=mode, restarts=1, steps=2, correction_steps=3
        )
        batch = torch.tensor([0, 1], device=device)
        yield (
            f"AtomicColumns-{mode}",
            optimizer,
            list(model.parameters()),
            lambda opt=optimizer, b=batch: opt.step(b),
        )
