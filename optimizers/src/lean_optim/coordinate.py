# Copyright (c) 2026 Mikhail Borisov
"""Coordinate-wise adaptive rules: AMSGradW, AdamW, Adam, AdamX, AdaGrad, AdamNC and RMSProp.

Ports of the rules of `optimizer_benchmark.coordinate.CoordinateOptimizer`
and of `magma_benchmark.optimizer.RMSPropOptimizer`,
with the arithmetic of the historical trainers. A coefficient that stays
constant is a Python number, applied as the modular trainer applied it:
`m.mul_(b).add_(g, alpha=1 - b)`. A coefficient that changes with the update
is a float64 tensor on the parameter's device, computed from the update count
`state["step"]` and applied as `full_compile_benchmark.optimizer.coordinate`
applied it: `m.mul_(b).add_(g * (1 - b))`, then `c_t * m / denominator`.
A captured update therefore reads the count it advances, and replays.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

from . import config as optimizers
from ._parameters import MATRIX_NDIM
from .direction import DirectionOptimizer

if TYPE_CHECKING:
    from ._types import CoordinateSpec, ParameterModel, Scalar, StateDict, TensorState
    from ._workspace import Workspace
    from .config import Decay


def factor(decay: Decay, step: Tensor) -> Tensor | None:
    """c_t of `decay` at update `step`, a float64 tensor; None when it is constantly 1."""
    if isinstance(decay, optimizers.Constant):
        return None
    if isinstance(decay, optimizers.Inverse):
        return 1 / step
    if isinstance(decay, optimizers.InverseSqrt):
        return step.rsqrt()
    if isinstance(decay, optimizers.Geometric):
        return decay.ratio ** (step - 1)
    msg = f"No builder for {decay!r}"
    raise NotImplementedError(msg)


def scaled(value: float, decay: Decay, step: Tensor) -> Scalar:
    """Value c_t: the number itself while c_t is constantly 1."""
    c = factor(decay, step)
    return value if c is None else value * c


def average(m: Tensor, gradient: Tensor, beta: Scalar, work: Tensor | None = None) -> None:
    """M <- beta m + (1 - beta) g."""
    if torch.is_tensor(beta):
        if work is None:
            m.mul_(beta).add_(gradient * (1 - beta))
        else:
            torch.mul(gradient, 1 - beta, out=work)
            m.mul_(beta).add_(work)
    else:
        m.mul_(beta).add_(gradient, alpha=1 - beta)


def mean_square(v: Tensor, gradient: Tensor, step: Tensor, work: Tensor | None = None) -> None:
    """V <- ((t - 1) v + g^2) / t: the mean of the squared gradients."""
    if work is None:
        v.mul_((step - 1) / step).add_(gradient.square() / step)
    else:
        torch.square(gradient, out=work)
        work.div_(step)
        v.mul_((step - 1) / step).add_(work)


def ratio(numerator: Tensor, denominator: Tensor, c: Tensor | None) -> Tensor:
    """Normalize a numerator by the regularized second-moment square root."""
    return numerator / denominator if c is None else c * numerator / denominator


class Coordinate(DirectionOptimizer):
    """Buffers per parameter; `state["step"]` counts updates when a coefficient depends on it."""

    buffers = ()

    def __init__(
        self, spec: CoordinateSpec, model: ParameterModel, rate: Scalar, *, counted: bool
    ) -> None:
        """Initialize the parameters, hyperparameters and persistent state."""
        super().__init__(model, rate)
        self.spec, self.counted = spec, counted

    def begin(self, parameter: Tensor, state: TensorState) -> Tensor:
        """Initialize the parameter's state and count this update."""
        if not state:
            state.update({name: torch.zeros_like(parameter) for name in self.buffers})
            if self.counted:
                state["step"] = parameter.new_zeros((), dtype=torch.float64)
        if self.counted:
            state["step"].add_(1)
        return state

    def first_moment(self, parameter: Tensor) -> str | None:  # noqa: ARG002 - Implement the shared optimizer interface.
        """Return the raw first-moment state key, if the rule keeps one."""
        return "m" if "m" in self.buffers else None

    def average(self, parameter: Tensor, m: Tensor, gradient: Tensor, beta: Scalar) -> None:
        """Update the raw first moment without a full-size temporary allocation."""
        if torch.is_tensor(beta):
            work = self.workspace.like(parameter, "work")
            complement = torch.neg(beta, out=self.workspace.like(beta, "complement"))
            complement.add_(1)
            torch.mul(gradient, complement, out=work)
            m.mul_(beta).add_(work)
        else:
            # Preserve the historical scalar EMA and its diagnostic override.
            average(m, gradient, beta)

    def factor(self, decay: Decay, step: Tensor, name: str = "factor") -> Tensor | None:
        """Evaluate a scheduled coefficient in a reusable scalar buffer."""
        if isinstance(decay, optimizers.Constant):
            return None
        output = self.workspace.like(step, name)
        if isinstance(decay, optimizers.Inverse):
            return torch.reciprocal(step, out=output)
        if isinstance(decay, optimizers.InverseSqrt):
            return torch.rsqrt(step, out=output)
        if isinstance(decay, optimizers.Geometric):
            torch.sub(step, 1, out=output)
            base = self.workspace.like(output, "base").fill_(decay.ratio)
            return torch.pow(base, output, out=output)
        msg = f"No builder for {decay!r}"
        raise NotImplementedError(msg)

    def scaled(self, value: float, decay: Decay, step: Tensor, name: str = "scaled") -> Scalar:
        """Multiply a coefficient by its schedule value."""
        coefficient = self.factor(decay, step, name)
        return value if coefficient is None else coefficient.mul_(value)

    def mean_square(self, parameter: Tensor, v: Tensor, gradient: Tensor, step: Tensor) -> None:
        """Update the mean squared gradient using a reusable work tensor."""
        work = self.workspace.like(parameter, "work")
        coefficient = torch.sub(step, 1, out=self.workspace.like(step, "mean_coefficient"))
        coefficient.div_(step)
        torch.square(gradient, out=work)
        work.div_(step)
        v.mul_(coefficient).add_(work)

    def ratio(
        self,
        parameter: Tensor,
        numerator: Tensor,
        second: Tensor,
        coefficient: Tensor | None = None,
    ) -> Tensor:
        """Normalize a numerator by the regularized second-moment square root."""
        output = self.workspace.like(parameter, "direction")
        torch.sqrt(second, out=output)
        output.add_(self.spec.eps)
        if coefficient is not None:
            work = self.workspace.like(parameter, "work")
            torch.mul(numerator, coefficient, out=work)
            numerator = work
        return torch.div(numerator, output, out=output)

    def decay(
        self, parameter: Tensor, direction: Tensor, coefficient: Tensor | None = None
    ) -> Tensor:
        """Add independent decay at the current parameter value."""
        if self.spec.weight_decay:
            work = self.workspace.like(parameter, "work")
            torch.mul(parameter, self.spec.weight_decay, out=work)
            if coefficient is not None:
                work.mul_(coefficient)
            direction.add_(work)
        return direction

    def load_state_dict(self, state_dict: StateDict) -> None:
        """Restore mathematical state and discard temporary work buffers."""
        super().load_state_dict(state_dict)
        # A restored nonconstant schedule needs its existing update clock,
        # even if this instance was constructed with constant coefficients.
        self.counted = isinstance(self, (AdamW, AdaGrad, AdamNC)) or varies(
            *(
                getattr(self.spec, name)
                for name in ("beta1_decay", "lr_decay")
                if name in self.param_groups[0]
            )
        )


def varies(*decays: Decay) -> bool:
    """Return whether any coefficient schedule changes with the update count."""
    return any(not isinstance(decay, optimizers.Constant) for decay in decays)


class AMSGradW(Coordinate):
    """The `amsgrad` rule on raw moments, plus `weight_decay` times the parameter."""

    buffers = ("m", "v", "maximum")
    moments = "raw_buffers_without_bias_correction_for_AMSGradW"

    def __init__(
        self, spec: CoordinateSpec, model: ParameterModel, rate: Scalar | None = None
    ) -> None:
        """Initialize the parameters, hyperparameters and persistent state."""
        super().__init__(spec, model, rate, counted=varies(spec.beta1_decay, spec.lr_decay))

    def direction(self, parameter: Tensor, state: TensorState) -> Tensor:
        """Compute the update direction from the current gradient and state."""
        spec, gradient, state = self.spec, parameter.grad, self.begin(parameter, state)
        step = state.get("step")
        (beta, beta2), m, v, maximum = spec.betas, state["m"], state["v"], state["maximum"]
        self.average(parameter, m, gradient, self.scaled(beta, spec.beta1_decay, step))
        v.mul_(beta2).addcmul_(gradient, gradient, value=1 - beta2)
        torch.maximum(maximum, v, out=maximum)
        coefficient = self.factor(spec.lr_decay, step)
        direction = self.ratio(parameter, m, maximum, coefficient)
        return self.decay(parameter, direction, coefficient)


class AdamW(Coordinate):
    """The `adamw` rule: PyTorch's AdamW as a direction, for stages to transform.

    m / (1 - b1^t) over sqrt(v / (1 - b2^t)) + eps, plus `weight_decay` times
    each decayed parameter, which the update multiplies by the rate as
    PyTorch's decoupled decay does. It parts from the native AdamW in the last
    bits.
    """

    buffers = ("m", "v")
    moments = "raw_buffers_before_bias_correction_for_AdamW"

    def __init__(
        self, spec: CoordinateSpec, model: ParameterModel, rate: Scalar | None = None
    ) -> None:
        """Initialize the parameters, hyperparameters and persistent state."""
        super().__init__(spec, model, rate, counted=True)

    def direction(self, parameter: Tensor, state: TensorState) -> Tensor:
        """Compute the update direction from the current gradient and state."""
        spec, gradient, state = self.spec, parameter.grad, self.begin(parameter, state)
        (beta, beta2), m, v, step = spec.betas, state["m"], state["v"], state["step"]
        average(m, gradient, beta)
        v.mul_(beta2).addcmul_(gradient, gradient, value=1 - beta2)
        direction = self.workspace.like(parameter, "direction")
        work = self.workspace.like(parameter, "work")
        torch.div(m, bias_correction(beta, step, self.workspace, "bias1"), out=work)
        torch.div(v, bias_correction(beta2, step, self.workspace, "bias2"), out=direction)
        direction.sqrt_().add_(spec.eps)
        torch.div(work, direction, out=direction)
        if spec.weight_decay and (spec.decay == "all" or parameter.ndim >= MATRIX_NDIM):
            self.decay(parameter, direction)
        return direction


class Adam(Coordinate):
    """The `adam` rule: raw moments, normalized by the second moment itself."""

    buffers = ("m", "v")
    moments = "raw_buffers_without_bias_correction_for_Adam"

    def __init__(
        self, spec: CoordinateSpec, model: ParameterModel, rate: Scalar | None = None
    ) -> None:
        """Initialize the parameters, hyperparameters and persistent state."""
        super().__init__(spec, model, rate, counted=varies(spec.beta1_decay, spec.lr_decay))

    def direction(self, parameter: Tensor, state: TensorState) -> Tensor:
        """Compute the update direction from the current gradient and state."""
        spec, gradient, state = self.spec, parameter.grad, self.begin(parameter, state)
        step = state.get("step")
        (beta, beta2), m, v = spec.betas, state["m"], state["v"]
        self.average(parameter, m, gradient, self.scaled(beta, spec.beta1_decay, step))
        v.mul_(beta2).addcmul_(gradient, gradient, value=1 - beta2)
        return self.ratio(parameter, m, v, self.factor(spec.lr_decay, step))


class AdamX(Coordinate):
    """The `adamx` rule: the running maximum rescaled by ((1 - b1_t) / (1 - b1_{t-1}))^2."""

    buffers = ("m", "v", "maximum")
    moments = "raw_buffers_without_bias_correction_for_AdamX"

    def __init__(
        self, spec: CoordinateSpec, model: ParameterModel, rate: Scalar | None = None
    ) -> None:
        """Initialize the parameters, hyperparameters and persistent state."""
        super().__init__(spec, model, rate, counted=varies(spec.beta1_decay, spec.lr_decay))

    def direction(self, parameter: Tensor, state: TensorState) -> Tensor:
        """Compute the update direction from the current gradient and state."""
        spec, gradient, state = self.spec, parameter.grad, self.begin(parameter, state)
        step = state.get("step")
        (beta, beta2), m, v, maximum = spec.betas, state["m"], state["v"], state["maximum"]
        current = self.scaled(beta, spec.beta1_decay, step)
        self.average(parameter, m, gradient, current)
        v.mul_(beta2).addcmul_(gradient, gradient, value=1 - beta2)
        if varies(spec.beta1_decay):
            # At the first update the rescaled maximum is zero whatever the factor.
            previous_step = torch.sub(step, 1, out=self.workspace.like(step, "previous_step"))
            previous_step.clamp_min_(1)
            previous = self.scaled(beta, spec.beta1_decay, previous_step, "previous")
            scale = self.workspace.like(step, "maximum_scale")
            torch.neg(current, out=scale)
            scale.add_(1)
            previous.neg_().add_(1)
            scale.div_(previous).square_()
            maximum.mul_(scale)
        torch.maximum(maximum, v, out=maximum)
        return self.ratio(parameter, m, maximum, self.factor(spec.lr_decay, step))


class AdaGrad(Coordinate):
    """The `adagrad` rule: the gradient over the root mean of the squared gradients."""

    buffers = ("v",)
    moments = "mean_squared_gradients_for_AdaGrad"

    def __init__(
        self, spec: CoordinateSpec, model: ParameterModel, rate: Scalar | None = None
    ) -> None:
        """Initialize the parameters, hyperparameters and persistent state."""
        super().__init__(spec, model, rate, counted=True)

    def direction(self, parameter: Tensor, state: TensorState) -> Tensor:
        """Compute the update direction from the current gradient and state."""
        spec, gradient, state = self.spec, parameter.grad, self.begin(parameter, state)
        self.mean_square(parameter, state["v"], gradient, state["step"])
        return self.ratio(
            parameter, gradient, state["v"], self.factor(spec.lr_decay, state["step"])
        )


class AdamNC(Coordinate):
    """The `adamnc` rule: momentum over the root mean of the squared gradients."""

    buffers = ("m", "v")
    moments = "raw_momentum_and_mean_squared_gradients_for_AdamNC"

    def __init__(
        self, spec: CoordinateSpec, model: ParameterModel, rate: Scalar | None = None
    ) -> None:
        """Initialize the parameters, hyperparameters and persistent state."""
        super().__init__(spec, model, rate, counted=True)

    def direction(self, parameter: Tensor, state: TensorState) -> Tensor:
        """Compute the update direction from the current gradient and state."""
        spec, gradient, state = self.spec, parameter.grad, self.begin(parameter, state)
        step, m, v = state["step"], state["m"], state["v"]
        self.mean_square(parameter, v, gradient, step)
        self.average(parameter, m, gradient, self.scaled(spec.beta1, spec.beta1_decay, step))
        return self.ratio(parameter, m, v, self.factor(spec.lr_decay, step))


class RMSProp(Coordinate):
    """The `rmsprop` rule: the gradient over the root of the raw second moment."""

    buffers = ("v",)
    moments = "raw_second_moment_for_RMSProp"

    def __init__(
        self, spec: CoordinateSpec, model: ParameterModel, rate: Scalar | None = None
    ) -> None:
        """Initialize the parameters, hyperparameters and persistent state."""
        super().__init__(spec, model, rate, counted=False)

    def direction(self, parameter: Tensor, state: TensorState) -> Tensor:
        """Compute the update direction from the current gradient and state."""
        spec, gradient, state = self.spec, parameter.grad, self.begin(parameter, state)
        state["v"].mul_(spec.beta2).addcmul_(gradient, gradient, value=1 - spec.beta2)
        return self.ratio(parameter, gradient, state["v"])


def bias_correction(beta: float, step: Tensor, workspace: Workspace, name: str) -> Tensor:
    """Evaluate one minus the exponential moment coefficient."""
    correction = workspace.like(step, name)
    base = workspace.like(correction, "base").fill_(beta)
    torch.pow(base, step, out=correction)
    return correction.neg_().add_(1)
