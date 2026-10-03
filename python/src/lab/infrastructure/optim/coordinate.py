"""Coordinate-wise adaptive rules: AMSGradW, Adam, AdamX, AdaGrad, AdamNC and RMSProp.

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

import torch

from ...domain import optimizers
from .direction import DirectionOptimizer


def factor(decay, step):
    """c_t of `decay` at update `step`, a float64 tensor; None when it is constantly 1."""
    if isinstance(decay, optimizers.Constant):
        return None
    if isinstance(decay, optimizers.Inverse):
        return 1 / step
    if isinstance(decay, optimizers.InverseSqrt):
        return step.rsqrt()
    if isinstance(decay, optimizers.Geometric):
        return decay.ratio ** (step - 1)
    raise NotImplementedError(f"No builder for {decay!r}")


def scaled(value, decay, step):
    """value c_t: the number itself while c_t is constantly 1."""
    c = factor(decay, step)
    return value if c is None else value * c


def average(m, gradient, beta):
    """m <- beta m + (1 - beta) g."""
    if torch.is_tensor(beta):
        m.mul_(beta).add_(gradient * (1 - beta))
    else:
        m.mul_(beta).add_(gradient, alpha=1 - beta)


def mean_square(v, gradient, step):
    """v <- ((t - 1) v + g^2) / t: the mean of the squared gradients."""
    v.mul_((step - 1) / step).add_(gradient.square() / step)


def ratio(numerator, denominator, c):
    return numerator / denominator if c is None else c * numerator / denominator


class Coordinate(DirectionOptimizer):
    """Buffers per parameter; `state["step"]` counts updates when a coefficient depends on it."""
    buffers = ()

    def __init__(self, spec, model, rate, counted):
        super().__init__(model, rate)
        self.spec, self.counted = spec, counted

    def begin(self, parameter, state):
        """The parameter's state, with this update counted."""
        if not state:
            state.update({name: torch.zeros_like(parameter) for name in self.buffers})
            if self.counted:
                state["step"] = parameter.new_zeros((), dtype=torch.float64)
        if self.counted:
            state["step"].add_(1)
        return state


def varies(*decays):
    return any(not isinstance(decay, optimizers.Constant) for decay in decays)


class AMSGradW(Coordinate):
    """The `amsgrad` rule on raw moments, plus `weight_decay` times the parameter."""
    buffers = ("m", "v", "maximum")
    moments = "raw_buffers_without_bias_correction_for_AMSGradW"

    def __init__(self, spec, model, rate=None):
        super().__init__(spec, model, rate, varies(spec.beta1_decay, spec.lr_decay))

    def direction(self, parameter, state):
        spec, gradient, state = self.spec, parameter.grad, self.begin(parameter, state)
        step = state.get("step")
        (beta, beta2), m, v, maximum = spec.betas, state["m"], state["v"], state["maximum"]
        average(m, gradient, scaled(beta, spec.beta1_decay, step))
        v.mul_(beta2).addcmul_(gradient, gradient, value=1 - beta2)
        torch.maximum(maximum, v, out=maximum)
        direction = ratio(m, maximum.sqrt() + spec.eps, factor(spec.lr_decay, step))
        if spec.weight_decay:
            direction = direction + spec.weight_decay * parameter
        return direction


class Adam(Coordinate):
    """The `adam` rule: raw moments, normalized by the second moment itself."""
    buffers = ("m", "v")
    moments = "raw_buffers_without_bias_correction_for_Adam"

    def __init__(self, spec, model, rate=None):
        super().__init__(spec, model, rate, varies(spec.beta1_decay, spec.lr_decay))

    def direction(self, parameter, state):
        spec, gradient, state = self.spec, parameter.grad, self.begin(parameter, state)
        step = state.get("step")
        (beta, beta2), m, v = spec.betas, state["m"], state["v"]
        average(m, gradient, scaled(beta, spec.beta1_decay, step))
        v.mul_(beta2).addcmul_(gradient, gradient, value=1 - beta2)
        return ratio(m, v.sqrt() + spec.eps, factor(spec.lr_decay, step))


class AdamX(Coordinate):
    """The `adamx` rule: the running maximum rescaled by ((1 - b1_t) / (1 - b1_{t-1}))^2."""
    buffers = ("m", "v", "maximum")
    moments = "raw_buffers_without_bias_correction_for_AdamX"

    def __init__(self, spec, model, rate=None):
        super().__init__(spec, model, rate, varies(spec.beta1_decay, spec.lr_decay))

    def direction(self, parameter, state):
        spec, gradient, state = self.spec, parameter.grad, self.begin(parameter, state)
        step = state.get("step")
        (beta, beta2), m, v, maximum = spec.betas, state["m"], state["v"], state["maximum"]
        current = scaled(beta, spec.beta1_decay, step)
        average(m, gradient, current)
        v.mul_(beta2).addcmul_(gradient, gradient, value=1 - beta2)
        if varies(spec.beta1_decay):
            # At the first update the rescaled maximum is zero whatever the factor.
            previous = scaled(beta, spec.beta1_decay, (step - 1).clamp_min(1))
            maximum.mul_(((1 - current) / (1 - previous)) ** 2)
        torch.maximum(maximum, v, out=maximum)
        return ratio(m, maximum.sqrt() + spec.eps, factor(spec.lr_decay, step))


class AdaGrad(Coordinate):
    """The `adagrad` rule: the gradient over the root mean of the squared gradients."""
    buffers = ("v",)
    moments = "mean_squared_gradients_for_AdaGrad"

    def __init__(self, spec, model, rate=None):
        super().__init__(spec, model, rate, True)

    def direction(self, parameter, state):
        spec, gradient, state = self.spec, parameter.grad, self.begin(parameter, state)
        mean_square(state["v"], gradient, state["step"])
        return ratio(gradient, state["v"].sqrt() + spec.eps, factor(spec.lr_decay, state["step"]))


class AdamNC(Coordinate):
    """The `adamnc` rule: momentum over the root mean of the squared gradients."""
    buffers = ("m", "v")
    moments = "raw_momentum_and_mean_squared_gradients_for_AdamNC"

    def __init__(self, spec, model, rate=None):
        super().__init__(spec, model, rate, True)

    def direction(self, parameter, state):
        spec, gradient, state = self.spec, parameter.grad, self.begin(parameter, state)
        step, m, v = state["step"], state["m"], state["v"]
        mean_square(v, gradient, step)
        average(m, gradient, scaled(spec.beta1, spec.beta1_decay, step))
        return ratio(m, v.sqrt() + spec.eps, factor(spec.lr_decay, step))


class RMSProp(Coordinate):
    """The `rmsprop` rule: the gradient over the root of the raw second moment."""
    buffers = ("v",)
    moments = "raw_second_moment_for_RMSProp"

    def __init__(self, spec, model, rate=None):
        super().__init__(spec, model, rate, False)

    def direction(self, parameter, state):
        spec, gradient, state = self.spec, parameter.grad, self.begin(parameter, state)
        state["v"].mul_(spec.beta2).addcmul_(gradient, gradient, value=1 - spec.beta2)
        return gradient / (state["v"].sqrt() + spec.eps)
