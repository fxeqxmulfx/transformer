"""Causal self-attention assembled from its blocks.

Projections yield head groups: FusedQKV one group holding every head on a
batch axis, PerHeadQKV one group per head. Scores, weights and XSA act on the
last two axes, so one code path serves both; each group reproduces the
operations of its historical source, which keeps both models bit-identical.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from ...domain import model
from .positions import rotate
from .sparsemax import causal_sparsemax


class FusedProjections(nn.Module):
    def __init__(self, width, heads, bias):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(width, 3 * width, bias=bias)

    def forward(self, x):
        batch, length, width = x.shape
        split = lambda t: t.view(batch, length, self.heads, width // self.heads).transpose(1, 2)
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        return [(split(q), split(k), split(v), None)]

    def merge(self, outputs):
        (z,) = outputs
        batch, heads, length, head = z.shape
        return z.transpose(1, 2).contiguous().view(batch, length, heads * head)


class Head(nn.Module):
    def __init__(self, width, head, bias):
        super().__init__()
        self.query = nn.Linear(width, head, bias=bias)
        self.key = nn.Linear(width, head, bias=bias)
        self.value = nn.Linear(width, head, bias=bias)


class PerHeadProjections(nn.Module):
    def __init__(self, width, heads, bias):
        super().__init__()
        self.heads = nn.ModuleList(Head(width, width // heads, bias) for _ in range(heads))

    def forward(self, x):
        return [(head.query(x), head.key(x), head.value(x), index) for index, head in enumerate(self.heads)]

    def merge(self, outputs):
        return torch.cat(outputs, dim=-1)


class ScaledDotScores(nn.Module):
    def __init__(self, head):
        super().__init__()
        self.scale = math.sqrt(head)

    def forward(self, q, k, rotary, index):
        if rotary is not None:
            q, k = rotate(q, *rotary), rotate(k, *rotary)
        return torch.matmul(q, k.transpose(-2, -1)) / self.scale


class QKNormScores(nn.Module):
    def __init__(self, heads, head, eps):
        super().__init__()
        self.eps = eps
        self.log_alpha = nn.Parameter(torch.full((heads,), 0.5 * math.log(head)))

    def forward(self, q, k, rotary, index):
        q = F.normalize(q, dim=-1, eps=self.eps)
        k = F.normalize(k, dim=-1, eps=self.eps)
        if rotary is not None:
            q, k = rotate(q, *rotary), rotate(k, *rotary)
        alpha = self.log_alpha.exp()
        alpha = alpha.view(1, -1, 1, 1) if index is None else alpha[index]
        return (q @ k.transpose(-2, -1)) * alpha


def softmax(scores):
    length = scores.shape[-1]
    future = torch.ones(length, length, device=scores.device, dtype=torch.bool).triu(1)
    return torch.softmax(scores.masked_fill(future, -torch.inf), dim=-1)


PROJECTIONS = {model.FusedQKV: FusedProjections, model.PerHeadQKV: PerHeadProjections}
WEIGHTS = {model.Softmax: softmax, model.Sparsemax: causal_sparsemax}


def scores_module(spec, heads, head):
    if isinstance(spec, model.ScaledDot):
        return ScaledDotScores(head)
    if isinstance(spec, model.QKNorm):
        return QKNormScores(heads, head, spec.eps)
    raise NotImplementedError(f"No builder for {spec!r}")


class Attention(nn.Module):
    def __init__(self, spec, width):
        super().__init__()
        head = width // spec.heads
        self.projections = PROJECTIONS[type(spec.projections)](width, spec.heads, spec.projections.bias)
        self.output = nn.Linear(width, width, bias=spec.output_bias)
        self.scores = scores_module(spec.scores, spec.heads, head)
        self.weights = WEIGHTS[type(spec.weights)]
        self.exclusive = None if spec.exclusive is None else spec.exclusive.eps

    def forward(self, x, rotary):
        outputs = []
        for q, k, v, index in self.projections(x):
            y = self.weights(self.scores(q, k, rotary, index)) @ v
            if self.exclusive is not None:
                v_hat = F.normalize(v, dim=-1, eps=self.exclusive)
                y = y - (y * v_hat).sum(dim=-1, keepdim=True) * v_hat
            outputs.append(y)
        return self.output(self.projections.merge(outputs))
