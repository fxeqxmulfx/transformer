"""Task162's isolated score-temperature candidate; see its domain specification.

The canonical attention graph at25f2ea22 is retained with one added query
multiplication. Gains are added after all common modules are initialized.
The current Q/K normalization, accurate RMS backward, RoPE and XSA stay intact.
"""

import torch
from torch import nn
from torch.nn import functional as F

from .adaptive_residual import CanonicalAttention, CanonicalResidual
from .centered_xsa import centered_projection, XSA_EPS
from .source_transformer import rotate


class TemperatureAttention(CanonicalAttention):
    def __init__(self, attention):
        super().__init__(attention)
        self.log_gain = nn.Parameter(torch.zeros(self.n_head))

    def forward(self, x, cos, sin):
        batch, length, width = x.shape
        q, k, v = self.c_attn(x).split(width, dim=2)
        q, k, v = (z.view(batch, length, self.n_head, self.head_dim).transpose(1, 2)
                   for z in (q, k, v))
        q, k = rotate(self.q_norm(q), cos, sin), rotate(self.k_norm(k), cos, sin)
        q = q * self.log_gain.exp().view(1, self.n_head, 1, 1)
        anchor = v[:, :, :1]
        average = F.scaled_dot_product_attention(q, k, v - anchor, is_causal=True)
        y = centered_projection(average, v - anchor, v)
        y[:, :, 0, :] = 0
        valid = (v[:, :, 0].norm(dim=-1) > XSA_EPS).all(dim=1)
        return self.c_proj(y.transpose(1, 2).contiguous().view(batch, length, -1)), valid


class TemperatureResidual(CanonicalResidual):
    def __init__(self, spec, vocab):
        super().__init__(spec, vocab)
        for block in self.blocks:
            block.sa = TemperatureAttention(block.sa)
