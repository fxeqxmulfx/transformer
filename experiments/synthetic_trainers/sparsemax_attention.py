"""Replace GPTMini causal softmax with sparsemax and preserve initialized parameters.

The simplex projection and support derivative are ported from this repository's
experiments/convex_mqar/src/convex_mqar/sparse_attention.py. The GPTMini forward
path retains QK normalization, learned temperatures, RoPE, XSA and projections.
Only row normalization changes; joint model training is not claimed convex.

The real-valued specification is Transformer.GPTMini.Convex.sparseWeights in
src/Transformer/GPTMini/Convex/Basic.lean. Its objective sum(a**2)/4 - sum(a*s)/2
has the same unique minimizer as ||a-s||**2 over a >= 0, sum(a) = 1, a[j>i] = 0.
Attention.lean: attentionWeights_projection_min connects this projection to
the original GPTMini scores. Scores retain their original scale. Lean defines
the exact minimizer; this floating-point solver and its derivative are checked
separately against exhaustive simplex-face KKT solutions and finite differences.
"""

import torch
from torch import nn
from torch.nn import functional as F

from experiments.gpt_mini import CausalMHA, GPTMini, apply_rope


class _Sparsemax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores):
        shifted = scores - scores.amax(-1, keepdim=True)
        ordered = shifted.sort(dim=-1, descending=True).values
        cumulative = ordered.cumsum(-1)
        ranks = torch.arange(1, scores.shape[-1] + 1, device=scores.device, dtype=scores.dtype)
        count = (1 + ranks * ordered > cumulative).sum(-1, keepdim=True)
        threshold = (cumulative.gather(-1, count - 1) - 1) / count
        weights = (shifted - threshold).clamp_min(0)
        ctx.save_for_backward(weights > 0)
        return weights

    @staticmethod
    def backward(ctx, gradient):
        (support,) = ctx.saved_tensors
        supported = gradient.masked_fill(~support, 0)
        average = supported.sum(-1, keepdim=True) / support.sum(-1, keepdim=True)
        return (gradient - average).masked_fill(~support, 0)


def causal_sparsemax(scores):
    """Project finite allowed scores onto each causal simplex; future mass is zero."""
    if scores.ndim < 2 or scores.shape[-2] != scores.shape[-1] or scores.shape[-1] == 0:
        raise ValueError("Causal sparsemax needs nonempty square score matrices")
    if not scores.is_floating_point():
        raise TypeError("Sparsemax scores must be floating point")
    length = scores.shape[-1]
    future = torch.ones(length, length, device=scores.device, dtype=torch.bool).triu(1)
    stable = scores if scores.dtype == torch.float64 else scores.float()
    return _Sparsemax.apply(stable.masked_fill(future, -torch.inf))


class SparsemaxMHA(CausalMHA):
    def __init__(self, original):
        nn.Module.__init__(self)
        self.n_heads, self.head_dim = original.n_heads, original.head_dim
        self.qkv, self.proj = original.qkv, original.proj
        self.log_alpha = original.log_alpha

    def forward(self, x, cos, sin):
        batch, length, width = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        q = F.normalize(q, dim=-1, eps=1e-6)
        k = F.normalize(k, dim=-1, eps=1e-6)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        alpha = self.log_alpha.exp().view(1, self.n_heads, 1, 1)
        scores = (q @ k.transpose(-2, -1)) * alpha
        attention = causal_sparsemax(scores)
        y = attention @ v
        v_hat = F.normalize(v, dim=-1, eps=1e-6)
        z = y - (y * v_hat).sum(dim=-1, keepdim=True) * v_hat
        return self.proj(z.transpose(1, 2).contiguous().view(batch, length, width))


def replace_attention(model):
    """Retain all parameter objects, names, tied weights and RNG state in place."""
    if type(model) is not GPTMini or any(type(block.attn) is not CausalMHA for block in model.blocks):
        raise ValueError("Sparsemax replacement requires the original GPTMini softmax layers")
    for block in model.blocks:
        block.attn = SparsemaxMHA(block.attn)
    return model
