"""Tensor-only compact inference for the two freely learned structured heads.

Source: Structured.TensorState/Pointer/Heads at b0a43a8. The head receives
only prenorm NTC tensors and globally shared weights. Deviations: stable
log-space prefix contractions; the all-pair sum is reassociated by key and
prefix value position. This keeps every causal key/value pair and all small
latent channels, without enumerating paths or the implicit 4**9 choices.
"""

import torch
from torch import nn
from torch.nn import functional as F


QUARTERS = ((1., 0.), (0., 1.), (-1., 0.), (0., -1.))


def recover(x):
    """Read genuine free fields and absolute positions after RMSNorm."""
    raw = x / x[..., 62:63]
    return raw[..., :52], raw[..., 63]


class TensorHeads(nn.Module):
    """Shared state/value, all-pair binding and branch parameters."""
    def __init__(self, context, std):
        super().__init__()
        self.context = context
        self.initial = nn.Parameter(torch.zeros(6))
        self.emission = nn.Parameter(torch.empty(6, 5, 4))
        nn.init.normal_(self.emission, std=std)
        self.branch = nn.Parameter(torch.zeros(2))
        self.chronology = nn.Parameter(torch.zeros(()))
        self.relative = nn.Parameter(torch.zeros(2 * context - 1))
        self.register_buffer("quarters", torch.tensor(QUARTERS))

    def state_channels(self, fields):
        """Every actual chronological endpoint marginal; no reference path."""
        batch, length = fields.shape[:2]
        transition = fields[..., :36].reshape(batch, length, 6, 6).softmax(-1)
        state = self.initial.softmax(-1).expand(batch, 6)
        history = []
        for position in range(length):
            state = torch.bmm(state[:, None, :], transition[:, position]).squeeze(1)
            history.append(state)
        marginal = torch.stack(history, 1)
        return torch.einsum("bts,shd->bthd", marginal, self.emission.softmax(-1))

    def pointer_terms(self, fields, positions):
        """Log partition of each real all-pair prefix, with no table mask."""
        batch, length = fields.shape[:2]
        query = fields[..., :16].reshape(batch, length, 4, 4)
        key = fields[..., 16:32].reshape(batch, length, 4, 4)
        value = fields[..., 32:52].reshape(batch, length, 5, 4)
        matching = torch.logsumexp(query[:, :, None] + key[:, None, :], -1).sum(-1)
        index = torch.arange(length, device=fields.device)
        relative = self.relative[index[None, :] - index[:, None] + self.context - 1]
        bias = positions[:, None, :] + self.chronology * index + relative
        routed = bias + torch.logsumexp(value, -1).sum(-1)[:, None, :]
        prefix = torch.logcumsumexp(routed, -1).transpose(1, 2)
        causal = index[None, :] <= index[:, None]
        logits = (matching + prefix).masked_fill(~causal, -torch.inf)
        return logits.logsumexp(-1), matching, routed, value, causal

    def pointer_channels(self, fields, positions):
        """True normalized inferred value-channel means for every prefix."""
        partition, matching, routed, value, causal = self.pointer_terms(fields, positions)
        # prefix[key, query, digit, channel] sums every visible value row.
        weighted = routed[..., None, None] + F.log_softmax(value, -1)[:, None]
        prefix = torch.logcumsumexp(weighted, 2).transpose(1, 2)
        logits = matching[..., None, None] + prefix
        logits = logits.masked_fill(~causal[None, :, :, None, None], -torch.inf)
        mass = torch.logsumexp(logits, 2) - partition[..., None, None]
        return mass.exp()

    def coordinates(self, fields, positions):
        """Ten axes of the actual mixed joint expectation."""
        state = self.state_channels(fields)
        pointer = self.pointer_channels(fields, positions)
        weights = self.branch.softmax(-1)
        mixed = weights[0] * state + weights[1] * pointer
        return torch.matmul(mixed, self.quarters).flatten(-2)

    def forward(self, x, rotary=None):
        """NTC attention interface; this replacement has learned positions."""
        if rotary is not None:
            raise ValueError("TensorHeads uses its learned physical positions, not RoPE")
        fields, positions = recover(x)
        return self.coordinates(fields, positions)
