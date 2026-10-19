"""Direction optimizers: every direction is computed before any parameter moves.

A port of `optimizer_benchmark.common.DirectionOptimizer`, with its
arithmetic: one group of all trainable parameters in `parameters()` order,
state created at the first update, and the update
`parameter.add_(direction, alpha=-lr)`. SGD is its plainest rule; the
adaptive ones are in `.coordinate`, the matrix ones in `.matrix`.

Stages (`.stages`) see the directions of all parameters together and return
the ones applied, as the descent guard does. A stage keeps its state
in the optimizer's `state` under its name, beside the rule's per-parameter
state, so checkpoints and graph warm-ups carry it like any other.

The capturable form reads the rate from a one-element device tensor when the
update runs, so a captured update replays with whatever the tensor holds; it
moves a parameter by `addcmul_(direction, rate, value=-1)` instead.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

from ._workspace import BufferedOptimizer

if TYPE_CHECKING:
    from ._types import (
        Closure,
        CoordinateSpec,
        Loss,
        ParameterModel,
        Report,
        Scalar,
        StateDict,
        TensorState,
    )


class DirectionOptimizer(BufferedOptimizer):
    # What the moment buffers are, for the diagnostics that record their norms.
    """Shared update protocol for DirectionOptimizer."""

    moments = None

    def __init__(self, model: ParameterModel, rate: Scalar) -> None:
        """Initialize the parameters, hyperparameters and persistent state."""
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        super().__init__(parameters, {"lr": 0.0 if rate is None else rate})
        self.capturable = rate is not None
        self.stages = []

    def add_param_group(self, param_group: StateDict) -> None:
        """Register the supported parameter group."""
        if self.param_groups:
            msg = "Joint-direction rules support one parameter group"
            raise ValueError(msg)
        super().add_param_group(param_group)

    def direction(self, parameter: Tensor, state: TensorState) -> Tensor:
        """Compute the update direction from the current gradient and state."""
        raise NotImplementedError

    def first_moment(self, parameter: Tensor) -> str | None:  # noqa: ARG002 - Implement the shared optimizer interface.
        """Return the raw first-moment state key, or None if the rule keeps none."""
        return

    def directions(self, parameters: list[Tensor]) -> list[Tensor]:
        """Compute every direction before moving any parameter."""
        return [self.direction(parameter, self.state[parameter]) for parameter in parameters]

    def report(self) -> Report:
        """Return each stage's counters by its name."""
        return {stage.name: stage.report(self.state[stage.name]) for stage in self.stages}

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        """Apply one update, evaluating an optional closure with gradients enabled."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        parameters = self.param_groups[0]["params"]
        if any(parameter.grad is None for parameter in parameters):
            parameters = [parameter for parameter in parameters if parameter.grad is not None]
        if not parameters:
            return loss
        if any(parameter.grad.is_sparse for parameter in parameters):
            msg = "These optimizers require dense gradients"
            raise RuntimeError(msg)
        if hasattr(self, "_clip_norm"):
            torch.nn.utils.clip_grad_norm_(parameters, self._clip_norm)
        directions = self.directions(parameters)
        for stage in self.stages:
            directions = stage(self, parameters, directions)
        rate = self.param_groups[0]["lr"]
        for parameter, direction in zip(parameters, directions, strict=True):
            if self.capturable:
                parameter.addcmul_(direction, rate, value=-1)
            else:
                parameter.add_(direction, alpha=-rate)
        return loss


class SGD(DirectionOptimizer):
    """The `sgd` rule: the gradient, plus `weight_decay` times the parameter."""

    def __init__(
        self, spec: CoordinateSpec, model: ParameterModel, rate: Scalar | None = None
    ) -> None:
        """Initialize the parameters, hyperparameters and persistent state."""
        super().__init__(model, rate)
        self.decay = spec.weight_decay

    def direction(self, parameter: Tensor, state: TensorState) -> Tensor:  # noqa: ARG002 - Implement the shared optimizer interface.
        """Compute the update direction from the current gradient and state."""
        decay = self.param_groups[0].get("weight_decay", self.decay)
        if decay:
            output = self.workspace.like(parameter, "direction")
            torch.mul(parameter, decay, out=output)
            return torch.add(parameter.grad, output, out=output)
        # Stages may overwrite their input; the plain update only reads it.
        return (
            parameter.grad
            if not self.stages
            else self.workspace.like(parameter, "direction").copy_(parameter.grad)
        )
