"""AdaFisher using genuine linear-layer activations and output derivatives.

FC factor sums, fresh-factor EMA weighting, separate min/max normalization,
damping, and positive-time raw-momentum correction match the corrected Lean
transition. The tied embedding uses its unembedding Linear factor; temperature
parameters use the source's identity-factor fallback. This shared-weight
curvature approximation is explicitly an experimental modeling choice.
"""

import torch
from torch import nn

from .common import DirectionOptimizer


def minmax(values):
    lower = values.amin()
    span = values.amax() - lower
    return (values - lower) / torch.where(span > 0, span, torch.ones_like(span))


class FactorCollector:
    def __init__(self, model):
        self.measurements = {}
        self.handles = []
        for module in model.modules():
            if isinstance(module, nn.Linear):
                if module.bias is not None:
                    raise ValueError("This GPTMini experiment uses bias-free Linear layers")
                self.handles.append(module.register_forward_hook(self.capture))

    def capture(self, module, inputs, output):
        if not module.training or not output.requires_grad:
            return
        with torch.no_grad():
            flattened = inputs[0].detach().reshape(-1, module.in_features)
            fresh_h = flattened.square().sum(dim=0)

        def backward(derivative):
            with torch.no_grad():
                fresh_s = derivative.detach().reshape(-1, module.out_features).square().sum(dim=0)
                if module.weight in self.measurements:
                    h, s = self.measurements[module.weight]
                    self.measurements[module.weight] = h + fresh_h, s + fresh_s
                else:
                    self.measurements[module.weight] = fresh_h, fresh_s
            return derivative

        output.register_hook(backward)

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.measurements.clear()


class AdaFisherOptimizer(DirectionOptimizer):
    def __init__(self, model, lr, *, beta=0.9, gamma=0.8, damping=0.001, decay=0.0):
        super().__init__(model.named_parameters(), lr)
        self.collector = FactorCollector(model)
        self.beta, self.gamma, self.damping, self.decay = beta, gamma, damping, decay

    def zero_grad(self, set_to_none=True):
        self.collector.measurements.clear()
        super().zero_grad(set_to_none=set_to_none)

    def propose(self, parameters):
        directions = []
        for p in parameters:
            rows, columns = p.shape if p.ndim == 2 else (p.numel(), 1)
            fresh = self.collector.measurements.get(p)
            if fresh is None:
                fresh = (torch.ones(columns, dtype=p.dtype, device=p.device),
                         torch.ones(rows, dtype=p.dtype, device=p.device))
            fresh_h, fresh_s = fresh
            state = self.state[p]
            if not state:
                state.update(h=torch.zeros_like(fresh_h), s=torch.zeros_like(fresh_s),
                             moment=torch.zeros_like(p))
            state["h"].mul_(1 - self.gamma).add_(fresh_h, alpha=self.gamma)
            state["s"].mul_(1 - self.gamma).add_(fresh_s, alpha=self.gamma)
            state["moment"].mul_(self.beta).add_(p.grad, alpha=1 - self.beta)
            metric = minmax(state["s"])[:, None] * minmax(state["h"])[None, :]
            metric = (metric + self.damping).reshape_as(p)
            corrected = state["moment"] / (1 - self.beta ** self.steps)
            direction = corrected / metric
            if self.decay:
                direction = direction + self.decay * p
            directions.append(direction)
        return directions

    def close(self):
        self.collector.close()
