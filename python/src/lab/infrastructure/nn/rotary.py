"""Tied Rotate90 residual classifiers, including the analytic Lean variants.

Adapted from grokking_abmod/web/own-channel/python/rotate90.py at
f49969393a7482490d45483d8a4fd3046b1adede (MIT; third_party/rotate90/NOTICE).
ReLU preserves its model and initializer. Deliberate extensions: configurable
prime and epsilon, SiLU, and three-projection SwiGLU with G variance 2/W,
U variance 1/W and D variance 1/H. PyTorch uses the lab's locked version;
its floating-point trajectories can differ from the browser and Rust.
Lean: Transformer.Landscape.Neural.SiLU and .SwiGLU at 00d36dd8.
"""

import torch
from torch import nn
from torch.nn import functional as F

from ...domain.rotary import DenseBranch, SwiGLU
from ..rotary_random import SplitMix64, normal_values
from .transformer import activation


class Rotate90(nn.Module):
    def __init__(self, spec, vocab, seed):
        super().__init__()
        if not 0 <= seed <= 0xFFFFFFFF:
            raise ValueError("The browser model seed is an unsigned 32-bit integer")
        self.spec, self.vocab = spec, vocab
        width = spec.width
        random = SplitMix64(seed ^ 0x73696D706C650004)

        def parameter(rows, columns, variance):
            values = normal_values(random, rows * columns, variance)
            return nn.Parameter(torch.tensor(values, dtype=torch.float32).reshape(rows, columns))

        self.E = parameter(vocab, width, 1 / width)
        if isinstance(spec.branch, DenseBranch):
            self.A = parameter(width, width, 2 / width)
            self.activation = activation(spec.branch.activation)
        elif isinstance(spec.branch, SwiGLU):
            hidden = spec.branch.hidden
            self.G = parameter(width, hidden, 2 / width)
            self.U = parameter(width, hidden, 1 / width)
            self.D = parameter(hidden, width, 1 / hidden)
        else:
            raise NotImplementedError(f"No Rotate90 branch for {spec.branch!r}")

    def normalize(self, x):
        return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + self.spec.eps)

    def forward(self, inputs):
        pairs = self.E[inputs[:, 1]].reshape(len(inputs), -1, 2)
        rotated = torch.stack((-pairs[..., 1], pairs[..., 0]), dim=-1).flatten(-2)
        x = self.E[inputs[:, 0]] + rotated
        r = self.normalize(x)
        branch = (self.activation(r @ self.A) if isinstance(self.spec.branch, DenseBranch)
                  else (F.silu(r @ self.G) * (r @ self.U)) @ self.D)
        return self.normalize(x + branch) @ self.E.T
