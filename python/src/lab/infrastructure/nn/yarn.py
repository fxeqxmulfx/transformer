"""YaRN frequency interpolation and derived causal temperature calibration.

Frequency source: arXiv:2309.00071v3, Section 3.2, Eqs. (10)-(13), with the
fixed-context rule of Section 3.4; Lean YaRN.yarnFrequency at 97e19a48.
Temperature source: Transformer.RoPE.YaRN.retentionGain and
calibrated_mass_lower_bound, src/Transformer/RoPE/YaRN/Calibration.lean,
4f4bbc61. This is an original numerical port of those definitions/proofs.
The adaptive choice of the current winner, per-row competitor count, and
finite-precision safeguards are implementation choices, not paper YaRN.
"""

import math

import torch

from .positions import rope_tables


def yarn_tables(head_width, context, original_context, theta):
    """Unit-length rotations; temperature is applied to the actual logits.

    The ramp is linear in rotation count, as manuscript Eq. (11), rather
    than a frequency-index ramp used by some vendor implementations.
    Contexts at or below the original return native tables bit for bit.
    """
    if context <= original_context:
        return rope_tables(head_width, context, theta)
    half = head_width // 2
    inverse = theta ** (-torch.arange(half, dtype=torch.float32) / half)
    scale = context / original_context
    rotations = original_context * inverse / (2 * math.pi)
    weight = ((rotations - 1) / 31).clamp(0, 1)
    adjusted = (1 - weight) * (inverse / scale) + weight * inverse
    angles = torch.outer(torch.arange(context, dtype=torch.float32), adjusted)
    return angles.cos(), angles.sin()


def retention_logits(scores, original_context):
    """Calibrate a square causal score matrix using its measured winner gap.

    For row i, M=i competitors and N=original_context-1. Where M>N and
    Delta=largest-visible-logit - second-largest-visible-logit is resolved,
    return g*(scores-max), g=1+log(M/N)/Delta. Centering is softmax invariant
    and avoids multiplying a large common offset. For equal-tail logits
    this exactly preserves 1/(1+N*exp(-Delta)); for arbitrary losing logits
    it preserves that lower bound, not the old row's actual probability.

    Deviations from the real-arithmetic theorem: preserve native scores
    on short rows and on gaps no larger than eps*max(1, visible score span),
    where eps is the working dtype's machine epsilon. Such rows have no
    retention certificate. This guard needs no tunable epsilon. Half and
    bfloat16 inputs are calibrated in float32. Future entries are zeroed
    before arithmetic, never affect the measured gap and are masked later.
    Gradients include the dependence of the gain on the measured gap;
    the resolved-gap decision and top-two selection are piecewise smooth.
    """
    if type(original_context) is not int or original_context < 2:
        raise ValueError("Original context must be an integer of at least two tokens")
    if scores.ndim < 2 or scores.shape[-2] != scores.shape[-1] or not scores.is_floating_point():
        raise ValueError("Retention needs a square floating-point causal score matrix")
    length = scores.shape[-1]
    if length <= original_context:
        return scores
    work = scores.float() if scores.dtype in (torch.float16, torch.bfloat16) else scores
    future = torch.ones(length, length, device=scores.device, dtype=torch.bool).triu(1)
    visible = work.masked_fill(future, -torch.inf)
    top = visible.topk(2, dim=-1).values
    maximum, runner_up = top[..., :1], top[..., 1:]
    gap = maximum - runner_up
    minimum = work.masked_fill(future, torch.inf).amin(dim=-1, keepdim=True)
    resolution = torch.finfo(work.dtype).eps * (maximum - minimum).clamp_min(1)
    competitors = torch.arange(length, dtype=work.dtype, device=scores.device)[:, None]
    growth = torch.log1p((competitors - (original_context - 1)).clamp_min(0) / (original_context - 1))
    active = (competitors > original_context - 1) & torch.isfinite(gap) & (gap > resolution)
    safe_gap = torch.where(active, gap, torch.ones_like(gap))
    centered = work.masked_fill(future, 0) - maximum
    calibrated = centered + growth * (centered / safe_gap)
    return torch.where(active, calibrated, work.masked_fill(future, 0))
