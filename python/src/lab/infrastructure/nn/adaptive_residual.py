"""Current source-equivalent GPT and conditional post-RMS amplitude candidate.

See domain.adaptive_residual for provenance and deviations. Common modules
are initialized before adding zero gate rows, without consuming random draws.
Every RMS, including per-head Q/K, uses the task155 separated backward.
"""

import torch
from torch import nn
from torch.nn import functional as F

from .centered_xsa import centered_projection, XSA_EPS
from .residual_transformer import ResidualTransformer
from .rms_backward import rms_norm
from .source_transformer import rotate


class AccurateRMSNorm(nn.RMSNorm):
    def forward(self, value):
        dtype = torch.float64 if value.dtype == torch.float64 else torch.float32
        eps = torch.finfo(dtype).eps if self.eps is None else self.eps
        return rms_norm(value, value.shape[-1], eps)


def accurate_norms(module):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.RMSNorm):
            module.add_module(name, AccurateRMSNorm(child.normalized_shape, eps=child.eps,
                                                    elementwise_affine=False))
        else:
            accurate_norms(child)


class CanonicalAttention(nn.Module):
    """Preserve df1e9f7's subtraction graph and first-position assignment.

    Sharing the two centered subtractions is equal over the reals but changes
    FP32 backward accumulation. Reuse the initialized modules without a draw.
    The lab interface omits cache support and always returns the prefix guard.
    """
    def __init__(self, attention):
        super().__init__()
        self.n_head, self.head_dim = attention.n_head, attention.head_dim
        for name, child in attention.named_children():
            self.add_module(name, child)

    def forward(self, x, cos, sin):
        batch, length, width = x.shape
        q, k, v = self.c_attn(x).split(width, dim=2)
        q, k, v = (z.view(batch, length, self.n_head, self.head_dim).transpose(1, 2)
                   for z in (q, k, v))
        q, k = rotate(self.q_norm(q), cos, sin), rotate(self.k_norm(k), cos, sin)
        anchor = v[:, :, :1]
        average = F.scaled_dot_product_attention(q, k, v - anchor, is_causal=True)
        y = centered_projection(average, v - anchor, v)
        y[:, :, 0, :] = 0
        valid = (v[:, :, 0].norm(dim=-1) > XSA_EPS).all(dim=1)
        return self.c_proj(y.transpose(1, 2).contiguous().view(batch, length, -1)), valid


class CanonicalResidual(ResidualTransformer):
    def __init__(self, spec, vocab):
        super().__init__(spec, vocab)
        for block in self.blocks:
            block.sa = CanonicalAttention(block.sa)
        accurate_norms(self)


class ResidualGain(nn.Linear):
    """Unit initial gain with no random draw and no trainable bias."""

    def reset_parameters(self):
        nn.init.zeros_(self.weight)

    def forward(self, value):
        return 2 * super().forward(value).sigmoid()


class AdaptiveBlock(nn.Module):
    def __init__(self, block, width):
        super().__init__()
        for name, child in block.named_children():
            self.add_module(name, child)
        self.attn_gain = ResidualGain(width, 1, bias=False)
        self.ffn_gain = ResidualGain(width, 1, bias=False)

    def attention(self, x, cos, sin):
        normalized = self.ln_1(x)
        value, valid = self.sa(normalized, cos, sin)
        return self.attn_gain(normalized) * self.attn_out_norm(value), valid

    def feed_forward(self, x):
        normalized = self.ln_2(x)
        return self.ffn_gain(normalized) * self.ffn_out_norm(self.ffwd(normalized))


class AdaptiveResidual(CanonicalResidual):
    def __init__(self, spec, vocab):
        super().__init__(spec, vocab)
        self.blocks = nn.ModuleList(AdaptiveBlock(block, spec.width) for block in self.blocks)
