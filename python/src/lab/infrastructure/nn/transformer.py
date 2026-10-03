"""The decoder-only transformer a `Transformer` spec describes.

Modules are constructed in the order of the historical models, so a seeded
build consumes the same random draws: embedding, then per block attention
and feed-forward, then the readout. Parameters are registered in that order
too, which keeps optimizer state dictionaries index-compatible.
"""

import torch
from torch import nn
from torch.nn import functional as F

from ...domain import model
from .attention import Attention
from .positions import rope_tables, sinusoid_table


class RMSNorm(nn.Module):
    def __init__(self, width, eps, scale):
        super().__init__()
        self.width, self.eps = width, eps
        self.weight = nn.Parameter(torch.ones(width)) if scale else None

    def forward(self, x):
        return F.rms_norm(x, (self.width,), self.weight, eps=self.eps)


def norm_module(spec, width):
    if isinstance(spec, model.RMSNorm):
        return RMSNorm(width, spec.eps, spec.scale)
    if isinstance(spec, model.LayerNorm):
        return nn.LayerNorm(width, eps=spec.eps, elementwise_affine=spec.affine)
    raise NotImplementedError(f"No builder for {spec!r}")


ACTIVATIONS = {model.ReLU: F.relu, model.ReLU2: lambda x: F.relu(x).square(), model.GELU: F.gelu}


class FeedForward(nn.Module):
    def __init__(self, spec, width):
        super().__init__()
        self.input = nn.Linear(width, spec.multiplier * width, bias=spec.bias)
        self.output = nn.Linear(spec.multiplier * width, width, bias=spec.bias)
        self.activation = ACTIVATIONS[type(spec.activation)]

    def forward(self, x):
        return self.output(self.activation(self.input(x)))


class Block(nn.Module):
    def __init__(self, spec, width):
        super().__init__()
        self.attention = Attention(spec.attention, width)
        self.attention_norm = norm_module(spec.norm, width)
        self.ffn = FeedForward(spec.ffn, width)
        self.ffn_norm = norm_module(spec.norm, width)
        self.pre_norm = isinstance(spec.residual, model.PreNorm)

    def forward(self, x, rotary):
        if self.pre_norm:
            x = x + self.attention(self.attention_norm(x), rotary)
            return x + self.ffn(self.ffn_norm(x))
        x = self.attention_norm(x + self.attention(x, rotary))
        return self.ffn_norm(x + self.ffn(x))


class Transformer(nn.Module):
    def __init__(self, spec, vocab):
        super().__init__()
        self.context = spec.context
        self.embed = nn.Embedding(vocab, spec.width)
        self.blocks = nn.ModuleList(Block(spec.block, spec.width) for _ in range(spec.depth))
        self.final_norm = None if spec.final_norm is None else norm_module(spec.final_norm, spec.width)
        bias = isinstance(spec.readout, model.Untied) and spec.readout.bias
        # A tied readout still draws its own matrix first, as GPTMini did.
        self.readout = nn.Linear(spec.width, vocab, bias=bias)
        if isinstance(spec.readout, model.Tied):
            self.readout.weight = self.embed.weight
        self.sinusoid = isinstance(spec.positions, model.Sinusoidal)
        self.rotary = isinstance(spec.positions, model.RoPE)
        if self.sinusoid:
            self.register_buffer("positions", sinusoid_table(spec.context, spec.width, spec.positions.base),
                                 persistent=False)
        if self.rotary:
            cos, sin = rope_tables(spec.head_width, spec.context, spec.positions.theta)
            self.register_buffer("cos", cos, persistent=False)
            self.register_buffer("sin", sin, persistent=False)

    def hidden_matrices(self):
        """The matrices of the blocks, attention and feed-forward weights, in parameter order."""
        return [parameter for parameter in self.blocks.parameters() if parameter.ndim == 2]

    def forward(self, tokens):
        length = tokens.shape[-1]
        if length > self.context:
            raise ValueError(f"Input of {length} tokens exceeds the context of {self.context}")
        x = self.embed(tokens)
        if self.sinusoid:
            x = x + self.positions[:length]
        rotary = (self.cos[:length], self.sin[:length]) if self.rotary else None
        for block in self.blocks:
            x = block(x, rotary)
        if self.final_norm is not None:
            x = self.final_norm(x)
        return self.readout(x)
