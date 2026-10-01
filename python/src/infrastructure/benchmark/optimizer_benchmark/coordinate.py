"""Diagonal rules read from AMSGrad and AdamBeyond Lean definitions.

Paper rules omit bias correction. Epsilon is a documented numerical extension;
projection is the identity on the unconstrained language-model parameter space.
"""

import math
import torch

from .common import DirectionOptimizer, moment_buffers


class CoordinateOptimizer(DirectionOptimizer):
    def __init__(self, named_parameters, lr, *, rule="amsgrad", schedule="constant",
                 beta=0.9, beta2=0.999, eps=1e-8, decay=0.0):
        super().__init__(named_parameters, lr)
        if rule not in {"sgd", "adagrad", "adam", "adamw", "amsgrad", "adamx", "adamnc"}:
            raise ValueError(rule)
        if schedule not in {"constant", "inverse", "geometric"}:
            raise ValueError(schedule)
        self.rule, self.schedule = rule, schedule
        self.beta, self.beta2, self.eps, self.decay = beta, beta2, eps, decay

    def beta_at(self, step):
        if self.schedule == "inverse":
            return self.beta / step
        if self.schedule == "geometric":
            return self.beta * 0.99 ** (step - 1)
        return self.beta

    def propose(self, parameters):
        step = self.steps
        beta = self.beta_at(step)
        decreasing = self.schedule != "constant" or self.rule in {"adagrad", "adamnc"}
        step_factor = 1 / math.sqrt(step) if decreasing else 1.0
        directions = []
        for p in parameters:
            g = p.grad
            state = self.state[p]
            m, v, maximum = moment_buffers(state, g)
            if self.rule == "sgd":
                direction = g.clone()
            elif self.rule in {"adagrad", "adamnc"}:
                v.mul_((step - 1) / step).addcmul_(g, g, value=1 / step)
                if self.rule == "adagrad":
                    m.copy_(g)
                else:
                    m.mul_(beta).add_(g, alpha=1 - beta)
                direction = step_factor * m / (v.sqrt() + self.eps)
            else:
                m.mul_(beta).add_(g, alpha=1 - beta)
                v.mul_(self.beta2).addcmul_(g, g, value=1 - self.beta2)
                if self.rule == "adamx":
                    if step > 1:
                        maximum.mul_(((1 - beta) / (1 - self.beta_at(step - 1))) ** 2)
                    torch.maximum(maximum, v, out=maximum)
                    denominator = maximum.sqrt() + self.eps
                elif self.rule == "amsgrad":
                    torch.maximum(maximum, v, out=maximum)
                    denominator = maximum.sqrt() + self.eps
                elif self.rule == "adamw":
                    denominator = (v / (1 - self.beta2 ** step)).sqrt() + self.eps
                else:
                    denominator = v.sqrt() + self.eps
                numerator = m / (1 - beta ** step) if self.rule == "adamw" else m
                direction = step_factor * numerator / denominator
            if self.decay:
                direction = direction + self.decay * p
            directions.append(direction)
        return directions
