"""AdaFisher: momentum over a damped, normalized Kronecker-factored Fisher diagonal.

A port of `optimizer_benchmark.fisher.AdaFisherOptimizer` with the arithmetic
of `full_compile_benchmark.optimizer`. The fresh factors of a linear layer are
collected by hooks while the update's forward and backward run: the input's
squares and the output gradient's squares, summed over positions. A weight
shared with a linear layer, as a tied embedding is with the readout, takes
that layer's factors; every other parameter takes ones.
"""

import torch
from torch import nn

from .direction import DirectionOptimizer


def minmax(values):
    lower = values.amin()
    span = values.amax() - lower
    return (values - lower) / torch.where(span > 0, span, torch.ones_like(span))


class Factors:
    """The fresh factors (h, s) of every linear weight in the last forward and backward of training."""

    def __init__(self, model):
        self.fresh = {}
        for module in model.modules():
            if isinstance(module, nn.Linear):
                if module.bias is not None:
                    raise NotImplementedError("AdaFisher's factors are collected for bias-free linear layers")
                module.register_forward_hook(self.capture)

    def capture(self, module, inputs, output):
        if not module.training or not output.requires_grad:
            return
        with torch.no_grad():
            h = inputs[0].detach().reshape(-1, module.in_features).square().sum(dim=0)

        def backward(derivative):
            with torch.no_grad():
                s = derivative.detach().reshape(-1, module.out_features).square().sum(dim=0)
                if module.weight in self.fresh:
                    earlier_h, earlier_s = self.fresh[module.weight]
                    self.fresh[module.weight] = earlier_h + h, earlier_s + s
                else:
                    self.fresh[module.weight] = h, s
            return derivative

        output.register_hook(backward)


class AdaFisher(DirectionOptimizer):
    """The `adafisher` rules; `weight_decay` 0.01 is `adafisherw`."""
    moments = "raw_first_moment_for_AdaFisher"

    def __init__(self, spec, model, rate=None):
        super().__init__(model, rate)
        self.spec, self.factors = spec, Factors(model)

    def zero_grad(self, set_to_none=True):
        self.factors.fresh.clear()
        super().zero_grad(set_to_none=set_to_none)

    def direction(self, parameter, state):
        spec, gradient = self.spec, parameter.grad
        rows, columns = parameter.shape if parameter.ndim == 2 else (parameter.numel(), 1)
        fresh = self.factors.fresh.get(parameter)
        h, s = fresh if fresh is not None else (parameter.new_ones(columns), parameter.new_ones(rows))
        if not state:
            state.update(h=parameter.new_zeros(columns), s=parameter.new_zeros(rows), m=torch.zeros_like(parameter),
                         step=parameter.new_zeros((), dtype=torch.float64))
        step = state["step"].add_(1)
        state["h"].mul_(1 - spec.gamma).add_(h, alpha=spec.gamma)
        state["s"].mul_(1 - spec.gamma).add_(s, alpha=spec.gamma)
        state["m"].mul_(spec.beta).add_(gradient, alpha=1 - spec.beta)
        fisher = (minmax(state["s"])[:, None] * minmax(state["h"])[None, :] + spec.damping).reshape_as(parameter)
        direction = (state["m"] / (1 - spec.beta ** step)) / fisher
        return direction + spec.weight_decay * parameter if spec.weight_decay else direction
