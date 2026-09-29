"""A new bounded-length convex recall variant with genuine standard RoPE.

Extension of 2211.11052v1 §3.1–3.2 and 2312.04927v1 app:synthetic, not a
paper theorem. Bipolar token bits occupy the last RoPE coordinate pairs;
each pair has one calibrated weight. Rotated distance is linear in those
weights. Lambda=0.1 and null cost=0.4 are chosen from bounds before evaluation.
The existing Lean result for unrotated codes does not cover this extension.
"""

import math

import torch
from torch import nn

from .convex import ConvexRecall


class RopeConvexRecall(nn.Module):
    def __init__(self, vocab, metric, width=64, base=10_000., max_length=512, num_pairs=None):
        super().__init__()
        if vocab < 2 or max_length < 4 or not math.isfinite(base) or base <= 1:
            raise ValueError("Use vocabulary >= 2, length >= 4, and a finite RoPE base > 1")
        self.vocab, self.width, self.base, self.max_length = vocab, width, base, max_length
        self.num_pairs = num_pairs
        self.bits = (vocab - 1).bit_length()
        self.pairs = (self.bits + 1) // 2
        self.quadratic, self.null_cost = 0.1, 0.4
        metric = torch.as_tensor(metric, dtype=torch.float32)
        if width % 2 or width < 2 * self.pairs or metric.shape != (self.pairs,):
            raise ValueError("Use an even RoPE width and one metric weight per active pair")
        if not torch.isfinite(metric).all() or not torch.all((metric >= 1) & (metric <= 1.000001)):
            raise ValueError("The calibrated metric must be within 1e-6 of the unit optimum")
        self.register_buffer("metric", metric)
        self.register_buffer("inverse_frequency", base ** (
            -torch.arange(0, width, 2).float() / width))
        bits = ((torch.arange(vocab)[:, None] >> torch.arange(self.bits)) & 1).float()
        codes = torch.zeros(vocab, width)
        offset = width - 2*self.pairs
        codes[:, offset:] = 1  # The unused final bit is fixed, not a token identifier.
        codes[:, offset:offset+self.bits] = 2*bits - 1
        self.register_buffer("codes", codes)
        weights = torch.zeros(width // 2)
        weights[-self.pairs:] = metric
        self.register_buffer("coordinate_weights", weights.repeat_interleave(2))
        self.margin = self.margin_bounds()
        if min(self.margin["match_null_gap"], self.margin["null_mismatch_gap"]) <= 2*self.quadratic + 1e-4:
            raise ValueError("This width/base/length combination does not meet the routing margin")

    def margin_bounds(self):
        """Sufficient real-arithmetic bounds; numbers are evaluated in float64.

        For equal signs a pair costs 1-cos(theta) <= theta²/2. A one-bit
        mismatch costs at least 1-|sin(theta)| >= 1-|theta|; a two-bit
        mismatch costs 1+cos(theta). Under |theta| < 1 the one-bit lower
        bound also bounds a two-bit mismatch. Other pair costs are nonnegative.
        """
        frequencies = self.inverse_frequency[-self.pairs:].double()
        delta = self.max_length - 1
        theta = delta * frequencies.max().item()
        if theta >= 1:
            raise ValueError("The bounded-angle margin requires a relative angle below one radian")
        match = self.metric.max().item() * delta**2 * frequencies.square().sum().item() / 2
        mismatch = self.metric.min().item() * (1 - theta)
        return {"max_relative_angle": theta, "match_cost_upper": match,
                "mismatch_cost_lower": mismatch, "null_cost": self.null_cost,
                "quadratic": self.quadratic, "required_gap": 2*self.quadratic,
                "match_null_gap": self.null_cost-match, "null_mismatch_gap": mismatch-self.null_cost,
                "scope": "Analytic bounds plus floating margin check; not a Lean proof"}

    def rotate(self, codes, positions):
        # Same interleaved RoPE frequencies and 2x2 rotations as RotaryAttention.
        angles = positions[..., None].float() * self.inverse_frequency
        cosine, sine = angles.cos(), angles.sin()
        even, odd = codes[..., 0::2], codes[..., 1::2]
        return torch.stack((even*cosine-odd*sine, even*sine+odd*cosine), -1).flatten(-2)

    def costs(self, tokens, positions):
        pairs = self.num_pairs or tokens.shape[1] // 4
        keys = tokens[:, :2*pairs:2]
        key_positions = torch.arange(0, 2*pairs, 2, device=tokens.device)[None, :]
        q = self.rotate(self.codes[tokens.gather(1, positions)], positions)
        k = self.rotate(self.codes[keys], key_positions)
        wq = q * self.coordinate_weights
        return ((wq*q).sum(-1, keepdim=True) +
                (k.square()*self.coordinate_weights).sum(-1)[:, None, :] -
                2*wq @ k.transpose(-1, -2)).clamp_min(0) / 4

    def routing_scores(self, tokens, positions):
        if tokens.shape[1] > self.max_length:
            raise ValueError("Sequence exceeds the declared RoPE margin domain")
        cost = self.costs(tokens, positions)
        pairs = cost.shape[-1]
        allowed = 2*torch.arange(pairs, device=tokens.device)[None, None, :] + 1 < positions[..., None]
        scores = (-cost / (2*self.quadratic)).masked_fill(~allowed, -torch.inf)
        null = torch.full_like(scores[..., :1], -self.null_cost / (2*self.quadratic))
        return torch.cat((scores, null), -1)

    def routing_weights(self, tokens, positions):
        return ConvexRecall.project_simplex(self.routing_scores(tokens, positions))

    def forward(self, tokens, positions):
        pairs = self.num_pairs or tokens.shape[1] // 4
        values = tokens[:, 1:2*pairs:2]
        weights = self.routing_weights(tokens, positions)
        logits = torch.zeros((*positions.shape, self.vocab), device=tokens.device)
        logits.scatter_add_(2, values[:, None, :].expand(-1, positions.shape[1], -1), weights[..., :pairs])
        prediction = logits.argmax(-1)
        return prediction.masked_fill(weights[..., -1] > logits.amax(-1), -1)


class RopeSoftmaxRecall(RopeConvexRecall):
    """Matched-feature control: same codes, metric, RoPE, mask, null and decoder."""

    def routing_weights(self, tokens, positions):
        return self.routing_scores(tokens, positions).softmax(-1)
