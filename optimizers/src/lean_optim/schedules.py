"""Warmup and constant, linear or cosine learning-rate schedules.

This is an explicit schedule extension of the package's Lean recurrences.
MD callbacks receive all independent rates multiplied by lr / initial_lr.
Warmup starts at zero; the update after warmup uses the full rate.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from torch import Tensor, optim

from ._checkpoint import rate_references, restore_rates

if TYPE_CHECKING:
    from ._types import StateDict


class LearningRateSchedule(optim.lr_scheduler.LRScheduler):
    """Warm up linearly, then keep or decay the rate, preserving device storage.

    Call step() after optimizer.step(), outside CUDA graph capture. Construct
    this scheduler before loading the optimizer and scheduler checkpoints.
    Linear/cosine decay reaches final_factor at total_steps completed updates
    and stays there. Their horizon must exceed warmup. Constant decay needs
    no horizon. Built-in warmup and outer warmup are alternative choices.
    Automatic-step methods and native Rprop have no supported outer LR schedule.
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        *,
        warmup: int = 0,
        total_steps: int | None = None,
        decay: str = "constant",
        final_factor: float = 0.0,
    ) -> None:
        """Validate compatibility and remember scalar peak rates before the zero start."""
        if not isinstance(optimizer, optim.Optimizer) or any(
            "lr" not in group for group in optimizer.param_groups
        ):
            msg = "LearningRateSchedule requires an optimizer with an outer learning rate"
            raise TypeError(msg)
        if isinstance(optimizer, optim.Rprop):
            msg = "Native Rprop uses lr only to initialize its adaptive step sizes"
            raise TypeError(msg)
        if type(warmup) is not int or warmup < 0:
            msg = "warmup must be a nonnegative integer"
            raise ValueError(msg)
        if decay not in {"constant", "linear", "cosine"}:
            msg = "decay must be constant, linear or cosine"
            raise ValueError(msg)
        if total_steps is not None and (type(total_steps) is not int or total_steps <= warmup):
            msg = "total_steps must be an integer greater than warmup"
            raise ValueError(msg)
        if decay != "constant" and total_steps is None:
            msg = "Linear and cosine decay require total_steps"
            raise ValueError(msg)
        if not math.isfinite(final_factor) or not 0 <= final_factor <= 1:
            msg = "final_factor must be finite and between zero and one"
            raise ValueError(msg)
        if warmup and any(group.get("warmup", 0) for group in optimizer.param_groups):
            msg = "Use built-in warmup or outer warmup, not both"
            raise ValueError(msg)
        self.warmup = warmup
        self.total_steps = total_steps
        self.decay = decay
        self.final_factor = final_factor
        self._base_values = [
            float(group.get("initial_lr", group["lr"])) for group in optimizer.param_groups
        ]
        if any(not math.isfinite(rate) or rate < 0 for rate in self._base_values):
            msg = "Peak learning rates must be finite and nonnegative"
            raise ValueError(msg)
        super().__init__(optimizer)
        self._rate_tensors = rate_references(optimizer)

    def get_lr(self) -> list[float]:
        """Return independent Python scalars without allocating CUDA tensors."""
        if self.last_epoch < self.warmup:
            factor = self.last_epoch / self.warmup
        elif self.total_steps is None or self.decay == "constant":
            factor = 1.0
        else:
            progress = min(1.0, (self.last_epoch - self.warmup) / (self.total_steps - self.warmup))
            remaining = (
                1 - progress if self.decay == "linear" else (1 + math.cos(math.pi * progress)) / 2
            )
            factor = self.final_factor + (1 - self.final_factor) * remaining
        return [rate * factor for rate in self._base_values]

    def get_last_lr(self) -> list[float]:
        """Return scalar rates without exposing mutable optimizer tensors."""
        return list(self._last_lr)

    def step(self, epoch: int | None = None) -> None:
        """Advance after an optimizer update, preserving any live device-rate storage."""
        if epoch is not None and (type(epoch) is not int or epoch < 0):
            msg = "The completed update count must be a nonnegative integer"
            raise ValueError(msg)
        self.last_epoch = self.last_epoch + 1 if epoch is None else epoch
        self._step_count += 1
        values = self.get_lr()
        for group, value in zip(self.optimizer.param_groups, values, strict=True):
            if isinstance(group["lr"], Tensor):
                group["lr"].fill_(value)
            else:
                group["lr"] = value
        self._last_lr = values

    def load_state_dict(self, state_dict: StateDict) -> None:
        """Restore counters and reconnect native optimizers to their destination rate tensors."""
        super().load_state_dict(state_dict)
        restore_rates(self.optimizer, self._rate_tensors)

    def state_dict(self) -> StateDict:
        """Serialize scheduling state without duplicating live optimizer rate references."""
        state = super().state_dict()
        state.pop("_rate_tensors")
        return state


class LinearWarmup(LearningRateSchedule):
    """Use min(1, completed / warmup), then keep the full rate."""

    def __init__(self, optimizer: optim.Optimizer, warmup: int) -> None:
        """Select constant learning rates after the requested linear warmup."""
        super().__init__(optimizer, warmup=warmup)
