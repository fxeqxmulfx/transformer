"""Replace only the learned RoPE Transformer's attention normalization.

Each causal weight row is the Euclidean projection of the scaled QK scores
onto the simplex. It solves min_a ||a||^2 / 2 - <a, scores>, subject to
a >= 0, sum(a) = 1, and a[j] = 0 for future positions. This has the same
minimizer as the row program in Transformer.GPTMini.Convex.Basic. It is a
convex inference problem; joint training of Q/K/V and the other layers is
not claimed to be convex. No dictionary parser or fixed token code is used.
"""

import math

import torch
from torch import nn

from .rope import RopeTransformer, RotaryAttention


class _Sparsemax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores):
        # Every causal row contains its diagonal, so at least one score is
        # finite. Subtracting the maximum preserves simplex projection and
        # avoids cancellation when all scores have a large common offset.
        shifted = scores - scores.amax(-1, keepdim=True)
        ordered = shifted.sort(dim=-1, descending=True).values
        cumulative = ordered.cumsum(-1)
        ranks = torch.arange(1, scores.shape[-1] + 1, device=scores.device,
                             dtype=scores.dtype)
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
    """Project square causal score rows; retain FP64 for derivative tests."""
    if scores.shape[-2] != scores.shape[-1]:
        raise ValueError("Causal attention requires square score matrices")
    length = scores.shape[-1]
    future = torch.ones(length, length, device=scores.device, dtype=torch.bool).triu(1)
    # BF16 thresholds can change the selected support or lose probability
    # mass. The projection and its reductions always use at least FP32.
    stable = scores if scores.dtype == torch.float64 else scores.float()
    return _Sparsemax.apply(stable.masked_fill(future, -torch.inf))


class SparsemaxAttention(RotaryAttention):
    def __init__(self, original):
        # Reuse the already initialized modules instead of drawing any new
        # weights. State names, parameter objects, and RNG state are retained.
        nn.Module.__init__(self)
        self.heads, self.head_width = original.heads, original.head_width
        self.qkv, self.output = original.qkv, original.output
        self.register_buffer("inverse_frequency", original.inverse_frequency,
                             persistent=False)

    def forward(self, x):
        batch, length, width = x.shape
        qkv = self.qkv(x).reshape(batch, length, 3, self.heads, self.head_width)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        q, k = self.rotate(q), self.rotate(k)
        # Explicit FP32 score formation and projection replace fused SDPA.
        # The numerical backend therefore differs, as well as normalization.
        with torch.autocast(x.device.type, enabled=False):
            score_dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
            scores = (q.to(score_dtype) @ k.to(score_dtype).transpose(-1, -2)
                      / math.sqrt(self.head_width))
            weights = causal_sparsemax(scores)
        attended = weights.to(v.dtype) @ v
        return self.output(attended.transpose(1, 2).reshape(batch, length, width))


class SparsemaxTransformer(RopeTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for block in self.blocks:
            block.attention = SparsemaxAttention(block.attention)
