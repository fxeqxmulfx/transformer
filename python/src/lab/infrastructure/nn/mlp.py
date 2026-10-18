"""The perceptron of `lab.domain.regression.MLP`.

Source: EXPERIMENT_PLAN.md, "Scalar MLP on mod 97 (2026-10-18)": a
perceptron reading x and y as two real numbers, or as one-hot vectors, and
answering with one. Its layers are constructed in the order of
`nn.Sequential(nn.Linear(d, w), act, nn.Linear(w, w), act, ..., nn.Linear(w, 1))`,
so the same seed draws the same parameters as that sequential model. It
deviates from it only in form: the layers sit in a `ModuleList`, `forward`
applies the activation between them, and the output is squeezed to one
number per row.
"""

import torch
from torch import nn
from torch.nn import functional as F

from ...domain.regression import OneHot, Scalar
from .transformer import activation


class MLP(nn.Module):
    def __init__(self, spec, vocab):
        super().__init__()
        self.spec, self.vocab = spec, vocab
        if not isinstance(spec.encoding, (Scalar, OneHot)):
            raise NotImplementedError(f"No builder for {spec.encoding!r}")
        inputs = spec.context * (vocab if isinstance(spec.encoding, OneHot) else 1)
        widths = [inputs] + [spec.width] * spec.depth
        self.hidden = nn.ModuleList(nn.Linear(fan_in, fan_out) for fan_in, fan_out in zip(widths, widths[1:]))
        self.output = nn.Linear(spec.width, 1)
        self.activation = activation(spec.activation)

    def encode(self, inputs):
        """Integer inputs of shape (rows, context) as the real rows the first layer reads."""
        if isinstance(self.spec.encoding, Scalar):
            return inputs.to(torch.float32) * (2 / (self.vocab - 1)) - 1
        return F.one_hot(inputs, self.vocab).flatten(1).to(torch.float32)

    def forward(self, inputs):
        hidden = self.encode(inputs)
        for layer in self.hidden:
            hidden = self.activation(layer(hidden))
        return self.output(hidden).squeeze(-1)
