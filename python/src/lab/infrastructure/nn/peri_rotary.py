"""Current Rotate90, ported from the source identified in domain.peri_rotary.

The sole interface change is a two-column input tensor instead of separate
a and b arguments. E is named consistently with the lab's observations.
Initialization preserves the source's SplitMix64 draw order and scales.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from ..rotary_random import SplitMix64, normal_values


class PeriRotate90(nn.Module):
    def __init__(self, spec, vocab, seed):
        super().__init__()
        self.spec, self.vocab = spec, vocab
        random = SplitMix64(seed ^ 0x73696D706C650004)

        def normal(rows, columns, deviation):
            values = normal_values(random, rows * columns, deviation ** 2)
            return nn.Parameter(torch.tensor(values, dtype=torch.float32).reshape(rows, columns))

        self.E = normal(vocab, spec.width, math.sqrt(1 / spec.width))
        self.gate_value, self.output = nn.ParameterList(), nn.ParameterList()
        for _ in range(spec.depth):
            self.gate_value.append(normal(spec.width, 2 * spec.hidden, 0.02))
            self.output.append(normal(spec.hidden, spec.width, 0.02 / math.sqrt(2 * spec.depth)))

    @staticmethod
    def normalize(x):
        return F.rms_norm(x, (x.size(-1),))

    def forward(self, inputs):
        pairs = self.E[inputs[:, 1]].unflatten(-1, (-1, 2))
        rotated = torch.stack((-pairs[..., 1], pairs[..., 0]), dim=-1).flatten(-2)
        x = self.normalize(self.E[inputs[:, 0]] + rotated)
        for gate_value, output in zip(self.gate_value, self.output):
            gate, value = (self.normalize(x) @ gate_value).chunk(2, dim=-1)
            x = x + self.normalize((F.silu(gate) * value) @ output)
        return self.normalize(x) @ self.E.T

    def hidden_matrices(self):
        return [*self.gate_value, *self.output]
