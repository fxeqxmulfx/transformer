"""Causal sparsemax with its support Jacobian.

Ported unchanged from `experiments/synthetic_trainers/sparsemax_attention.py`,
whose projection follows `experiments/convex_mqar/src/convex_mqar/
sparse_attention.py`. The real-valued specification is
Transformer.GPTMini.Convex.sparseWeights (src/Transformer/GPTMini/Convex/
Basic.lean); this floating-point solver is checked separately. No operation
reads a value back to the host, so the function is safe inside a CUDA graph.
"""

import torch


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
    """Project the visible scores of each row onto the simplex; future weights are zero."""
    if scores.ndim < 2 or scores.shape[-2] != scores.shape[-1] or scores.shape[-1] == 0:
        raise ValueError("Causal sparsemax needs nonempty square score matrices")
    if not scores.is_floating_point():
        raise TypeError("Sparsemax scores must be floating point")
    length = scores.shape[-1]
    future = torch.ones(length, length, device=scores.device, dtype=torch.bool).triu(1)
    stable = scores if scores.dtype == torch.float64 else scores.float()
    return _Sparsemax.apply(stable.masked_fill(future, -torch.inf))
