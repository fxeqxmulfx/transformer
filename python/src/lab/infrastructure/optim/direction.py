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

import torch


class DirectionOptimizer(torch.optim.Optimizer):
    # What the moment buffers are, for the diagnostics that record their norms.
    moments = None

    def __init__(self, model, rate):
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        super().__init__(parameters, {"lr": 0.0 if rate is None else rate})
        self.capturable = rate is not None
        self.stages = []

    def direction(self, parameter, state):
        raise NotImplementedError

    def first_moment(self, parameter):
        """The key of the rule's raw first moment of `parameter` in its state, or None if it keeps none."""
        return None

    def directions(self, parameters):
        """The direction of every parameter, before any parameter moves."""
        return [self.direction(parameter, self.state[parameter]) for parameter in parameters]

    def report(self):
        """What each stage counted, by its name."""
        return {stage.name: stage.report(self.state[stage.name]) for stage in self.stages}

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError("A direction optimizer updates on the gradients in place")
        parameters = [parameter for parameter in self.param_groups[0]["params"] if parameter.grad is not None]
        directions = self.directions(parameters)
        for stage in self.stages:
            directions = stage(self, parameters, directions)
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

