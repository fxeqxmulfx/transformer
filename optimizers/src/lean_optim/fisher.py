"""AdaFisher: momentum over a damped, normalized Kronecker-factored Fisher diagonal.

A port of `optimizer_benchmark.fisher.AdaFisherOptimizer` with the arithmetic
of `full_compile_benchmark.optimizer`. The fresh factors of a linear layer are
collected by hooks while the update's forward and backward run: the input's
squares and the output gradient's squares, summed over positions. A weight
shared with a linear layer, as a tied embedding is with the readout, takes
that layer's factors; every other parameter takes ones.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

from ._parameters import MATRIX_NDIM
from ._workspace import Workspace
from .coordinate import bias_correction
from .direction import DirectionOptimizer

if TYPE_CHECKING:
    from ._types import FisherSpec, Scalar, TensorState


def minmax(values: Tensor, workspace: Workspace | None = None) -> Tensor:
    """Normalize a nonconstant vector to the unit interval, using zero for a constant vector."""
    if workspace is None:
        lower = values.amin()
        span = values.amax() - lower
        return (values - lower) / torch.where(span > 0, span, torch.ones_like(span))
    lower = torch.amin(values, dim=None, out=workspace.tensor(values, "min", ()))
    span = torch.amax(values, dim=None, out=workspace.tensor(values, "span", ()))
    span.sub_(lower)
    output = torch.sub(values, lower, out=workspace.like(values, "normalized"))
    return output.div_(workspace.safe(span))


class Factors:
    """Collect fresh input/derivative factors of linear weights during training."""

    def __init__(self, model: torch.nn.Module) -> None:
        """Initialize the parameters, hyperparameters and persistent state."""
        self.fresh = {}
        self.handles = []
        self.workspace = {}
        modules = [module for module in model.modules() if isinstance(module, nn.Linear)]
        if any(module.bias is not None for module in modules):
            msg = "AdaFisher's factors are collected for bias-free linear layers"
            raise NotImplementedError(msg)
        try:
            for module in modules:
                if module.weight.requires_grad:
                    self.handles.append(module.register_forward_hook(self.capture))
        except Exception:
            for handle in self.handles:
                handle.remove()
            self.handles.clear()
            raise

    def _buffers(self, dtype: torch.dtype) -> Workspace:
        buffers = self.workspace.get(dtype)
        if buffers is None:
            buffers = self.workspace[dtype] = Workspace()
        return buffers

    def _factor(self, weight: Tensor, name: str, value: Tensor, previous: Tensor | None) -> Tensor:
        dtype = (
            value.dtype if previous is None else torch.promote_types(previous.dtype, value.dtype)
        )
        output = self._buffers(dtype).tensor(weight, name, value.shape, dtype)
        if previous is None:
            return output.copy_(value)
        return torch.add(previous, value, out=output)

    def capture(self, module: torch.nn.Linear, inputs: tuple[Tensor, ...], output: Tensor) -> None:
        """Collect input factors and register their paired derivative hook."""
        if not module.weight.requires_grad or not module.training or not output.requires_grad:
            return
        with torch.no_grad():
            squared = self._buffers(inputs[0].dtype).tensor(
                module.weight, "input.square", inputs[0].shape, inputs[0].dtype
            )
            torch.square(inputs[0].detach(), out=squared)
            # Each retained forward needs its own small h vector. Reusing it
            # would corrupt accumulated or repeated backward passes.
            h = squared.reshape(-1, module.in_features).sum(dim=0)

        def backward(derivative: Tensor) -> Tensor:
            """Accumulate derivative factors without changing the propagated gradient."""
            with torch.no_grad():
                buffers = self._buffers(derivative.dtype)
                squared = buffers.tensor(
                    module.weight, "derivative.square", derivative.shape, derivative.dtype
                )
                torch.square(derivative.detach(), out=squared)
                s = buffers.tensor(
                    module.weight, "derivative.sum", (module.out_features,), derivative.dtype
                )
                torch.sum(squared.reshape(-1, module.out_features), dim=0, out=s)
                previous = self.fresh.get(module.weight)
                # Accumulated forwards can cross autocast contexts. Preserve
                # ordinary addition's promotion instead of casting into the
                # first forward's low-precision buffer.
                fresh_h = self._factor(
                    module.weight, "fresh.h", h, None if previous is None else previous[0]
                )
                fresh_s = self._factor(
                    module.weight, "fresh.s", s, None if previous is None else previous[1]
                )
                self.fresh[module.weight] = fresh_h, fresh_s
            return derivative

        output.register_hook(backward)


class AdaFisher(DirectionOptimizer):
    """The `adafisher` rules; `weight_decay` 0.01 is `adafisherw`."""

    moments = "raw_first_moment_for_AdaFisher"

    def __init__(
        self, spec: FisherSpec, model: torch.nn.Module, rate: Scalar | None = None
    ) -> None:
        """Initialize the parameters, hyperparameters and persistent state."""
        super().__init__(model, rate)
        self.spec, self.factors = spec, Factors(model)

    def first_moment(self, parameter: Tensor) -> str | None:  # noqa: ARG002 - Implement the shared optimizer interface.
        """Return the raw first-moment state key, if the rule keeps one."""
        return "m"

    def zero_grad(self, set_to_none: bool = True) -> None:  # noqa: FBT001, FBT002 - Match torch.optim.Optimizer.zero_grad.
        """Clear fresh Fisher factors and reset gradients as PyTorch does."""
        self.factors.fresh.clear()
        super().zero_grad(set_to_none=set_to_none)

    def direction(self, parameter: Tensor, state: TensorState) -> Tensor:
        """Compute the update direction from the current gradient and state."""
        spec, gradient = self.spec, parameter.grad
        rows, columns = parameter.shape if parameter.ndim == MATRIX_NDIM else (parameter.numel(), 1)
        if not state:
            state.update(
                h=parameter.new_zeros(columns),
                s=parameter.new_zeros(rows),
                m=torch.zeros_like(parameter),
                step=parameter.new_zeros((), dtype=torch.float64),
            )
        fresh = self.factors.fresh.get(parameter)
        h, s = (
            fresh
            if fresh is not None
            else (
                self.workspace.like(state["h"], "identity").fill_(1),
                self.workspace.like(state["s"], "identity").fill_(1),
            )
        )
        step = state["step"].add_(1)
        state["h"].mul_(1 - spec.gamma).add_(h, alpha=spec.gamma)
        state["s"].mul_(1 - spec.gamma).add_(s, alpha=spec.gamma)
        state["m"].mul_(spec.beta).add_(gradient, alpha=1 - spec.beta)
        fisher = self.workspace.like(parameter, "fisher")
        torch.mul(
            minmax(state["s"], self.workspace)[:, None],
            minmax(state["h"], self.workspace)[None, :],
            out=fisher.view(rows, columns),
        )
        fisher.add_(spec.damping)
        direction = self.workspace.like(parameter, "direction")
        torch.div(
            state["m"], bias_correction(spec.beta, step, self.workspace, "bias"), out=direction
        )
        direction.div_(fisher)
        if spec.weight_decay:
            torch.mul(parameter, spec.weight_decay, out=fisher)
            direction.add_(fisher)
        return direction
