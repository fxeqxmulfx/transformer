"""Ports of diff_gpt attention_residual and conditioned_attention_residual.

Source: 0fab228, Lean AttentionResiduals.QueryCoordinates QC1--QC5;
key-only scoring/zero initialization from arXiv:2603.15031v1 §3.1/§5.
The single module selects a declared formula; no learned temperature or
value normalization is added. Finite epsilon is preserved under conditioning.
"""

import torch
from torch import nn
from torch.nn import functional as F


class DepthMixer(nn.Module):
    def __init__(self, width, mixing):
        super().__init__()
        self.width, self.mixing, self.eps = width, mixing, 1e-6
        self.query = nn.Parameter(torch.empty(width))
        with torch.no_grad():
            self.query.normal_()
            self.query.div_(self.query.square().mean().sqrt().clamp_min(1e-12))
            if mixing == "attnres":
                self.query.zero_()
        if mixing == "attnres":
            self.key_gain = nn.Parameter(torch.ones(width))
        if mixing == "conditioned_qk":
            self.register_buffer("query_scale", self.query.new_ones(()))

    def forward(self, values):
        if self.mixing == "attnres":
            keys = F.rms_norm(values, (self.width,), self.key_gain, eps=self.eps)
            scores = (keys * self.query).sum(-1)
        else:
            if self.mixing == "conditioned_qk":
                eps = self.eps / self.query_scale / self.query_scale
                query = self.query * torch.rsqrt(self.query.square().mean() + eps)
            else:
                query = F.rms_norm(self.query, (self.width,), eps=self.eps)
            keys = F.rms_norm(values, (self.width,), eps=self.eps)
            scores = (keys * query).sum(-1) / self.width**.5
        return (scores.softmax(-1)[..., None] * values).sum(-2)
