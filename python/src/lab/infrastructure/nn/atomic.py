"""Actual causal sparsemax heads and finite probability mixtures.

Source: matchingHeadScores, matchingHeadOutput and matchingMixtureSample
at 83d985f. The only numerical deviations are floating-point arithmetic and
finite atom storage. Selecting readout rows skips only unused value products.
PairedMatching implements pairedHeadLift/pairedHeadOutput using factorized
current/preceding token lookups rather than allocating the expanded pair
vocabulary in PairedMatchingHead.lean; its numerical forward is the same.
"""

import torch
from torch import nn

from ...domain.atomic import PairedMatching
from .sparsemax import causal_sparsemax


def head_forward(q, k, values, tokens, readout_positions=None):
    """One shared physical Q/K/value table; no inverse, normalizer or surrogate."""
    return _response(head_scores(q, k, tokens), values, tokens, readout_positions)


def head_scores(q, k, tokens):
    return q[tokens] @ k[tokens].transpose(-1, -2)


def paired_head_forward(q, k, values, tokens, readout_positions=None):
    """pairedHeadOutput: factorized current/preceding key roles, independent original values."""
    return _response(paired_head_scores(q, k, tokens), values, tokens, readout_positions)


def paired_head_scores(q, k, tokens):
    if q.shape[-1] % 2:
        raise ValueError("Paired matching needs an even Q/K width")
    half = q.shape[-1] // 2
    preceding = torch.cat((tokens[:, :1], tokens[:, :-1]), dim=1)
    keys = torch.cat((k[tokens][..., :half], k[preceding][..., half:]), dim=-1)
    return q[tokens] @ keys.transpose(-1, -2)


def _response(scores, values, tokens, readout_positions):
    weights = causal_sparsemax(scores).to(values.dtype)
    if readout_positions is not None:
        weights = weights.gather(1, readout_positions.unsqueeze(-1).expand(-1, -1, tokens.shape[1]))
    return weights @ values[tokens]


class MatchingHead(nn.Module):
    """A freely searched atom, with original independently stored values."""

    def __init__(self, q, k, values, forward_map=head_forward):
        super().__init__()
        self.q, self.k, self.values = (nn.Parameter(t.clone()) for t in (q, k, values))
        self.forward_map = forward_map

    def forward(self, tokens, readout_positions=None):
        return self.forward_map(self.q, self.k, self.values, tokens, readout_positions)


class AtomicMatching(nn.Module):
    """Selected bounded atoms and simplex mass, in checkpoint-stable storage."""

    def __init__(self, spec, vocab):
        super().__init__()
        self.spec = spec
        self.forward_map = paired_head_forward if isinstance(spec, PairedMatching) else head_forward
        self.score_map = paired_head_scores if isinstance(spec, PairedMatching) else head_scores
        channels = vocab if spec.channels is None else spec.channels
        dtype = getattr(torch, spec.precision)
        self.q = nn.Parameter(torch.zeros(spec.heads, vocab, spec.width, dtype=dtype))
        self.k = nn.Parameter(torch.zeros_like(self.q))
        self.values = nn.Parameter(torch.zeros(spec.heads, vocab, channels, dtype=dtype))
        with torch.no_grad():
            self.q[0].fill_(spec.initial_query)
            self.k[0].copy_(torch.linspace(-spec.initial_key_spread, spec.initial_key_spread,
                                          vocab, dtype=dtype).unsqueeze(1).expand(-1, spec.width))
            self.values[0].fill_(spec.initial_value)
        self.register_buffer("mass", torch.zeros(spec.heads, dtype=dtype))
        self.mass[0] = 1
        self.register_buffer("active", torch.tensor(1))

    def forward(self, tokens, readout_positions=None):
        result = None
        for index in range(int(self.active)):
            response = self.forward_map(self.q[index], self.k[index], self.values[index], tokens, readout_positions)
            weighted = self.mass[index] * response
            result = weighted if result is None else result + weighted
        return result

    @torch.no_grad()
    def append(self, head):
        index = int(self.active)
        if index == self.spec.heads:
            raise ValueError("AtomicMatching storage is full")
        for name in ("q", "k", "values"):
            getattr(self, name)[index].copy_(getattr(head, name))
        self.mass[index] = 0
        self.active.add_(1)

    @torch.no_grad()
    def prune(self):
        """Remove exactly zero masses; no approximate prediction compression."""
        count = int(self.active)
        keep = (self.mass[:count] > 0).nonzero().flatten()
        for tensor in (self.q, self.k, self.values, self.mass):
            tensor[:len(keep)].copy_(tensor[:count].index_select(0, keep))
            tensor[len(keep):count].zero_()
        self.active.fill_(len(keep))

    @torch.no_grad()
    def inspect(self):
        count = int(self.active)
        return {"active_heads": count, "mass": self.mass[:count].tolist(),
                "mass_sum": float(self.mass.sum()), "minimum_mass": float(self.mass.min()),
                "maximum_coordinate": max(float(t[:count].abs().max()) for t in (self.q, self.k, self.values)),
                "stored_scalars": sum(t.numel() for t in (self.q, self.k, self.values, self.mass)),
                "active_scalars": count * (self.q[0].numel() * 2 + self.values[0].numel() + 1)}
