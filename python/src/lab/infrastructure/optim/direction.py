"""Direction optimizers: every direction is computed before any parameter moves.

A port of `optimizer_benchmark.common.DirectionOptimizer`, with its
arithmetic: one group of all trainable parameters in `parameters()` order,
state created at the first update, and the update
`parameter.add_(direction, alpha=-lr)`. SGD is its plainest rule; the
adaptive ones are in `.coordinate`.

The capturable form reads the rate from a one-element device tensor when the
update runs, so a captured update replays with whatever the tensor holds; it
moves a parameter by `addcmul_(direction, rate, value=-1)` instead.
"""

import torch


class DirectionOptimizer(torch.optim.Optimizer):
    # What the moment buffers are, for the diagnostics that record their norms.
    moments = None

    def __init__(self, model, rate):
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        super().__init__(parameters, {"lr": 0.0 if rate is None else rate})
        self.capturable = rate is not None

    def direction(self, parameter, state):
        raise NotImplementedError

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError("A direction optimizer updates on the gradients in place")
        parameters = [parameter for parameter in self.param_groups[0]["params"] if parameter.grad is not None]
        directions = [self.direction(parameter, self.state[parameter]) for parameter in parameters]
        rate = self.param_groups[0]["lr"]
        for parameter, direction in zip(parameters, directions, strict=True):
            if self.capturable:
                parameter.addcmul_(direction, rate, value=-1)
            else:
                parameter.add_(direction, alpha=-rate)


class SGD(DirectionOptimizer):
    """The `sgd` rule: the gradient, plus `weight_decay` times the parameter."""

    def __init__(self, spec, model, rate=None):
        super().__init__(model, rate)
        self.decay = spec.weight_decay

    def direction(self, parameter, state):
        direction = parameter.grad.clone()
        if self.decay:
            direction = direction + self.decay * parameter
        return direction

