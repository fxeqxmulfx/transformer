"""Recurrences of domain.gradient_paths; no learned feature extractor.

For fixed linear branches use (1+c)*state at each layer. This is the exact
real identity proved by Residual.residualRun_linear; it avoids Inductor
recursively recounting both uses of state in state+c*state (task151 probe).
The FP32 rounding order changes; independent values/derivatives are checked.
"""

import torch
from torch import nn
from torch.nn import functional as F

from ...domain.gradient_paths import AccurateBranchGates, BranchGates, ContextGates
from .context_gates import ContextGatedBranches
from .rms_backward import rms_norm


class GatedBranches(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.spec = spec
        self.logits = nn.Parameter(torch.zeros(spec.depth))

    def normalize(self, value):
        return F.rms_norm(value, (self.spec.width,), eps=self.spec.eps)

    def forward(self, inputs):
        gains = (2 * self.logits.sigmoid())[None, :, None]
        branches = self.spec.scale * inputs[:, 1:]
        if self.spec.placement == "before":
            branches = self.normalize(gains * branches)
        else:
            branches = gains * self.normalize(branches)
        return inputs[:, 0] + branches.sum(dim=1)

    def coefficients(self):
        gain = 2 * self.logits.sigmoid()
        if self.spec.placement == "before":
            amplitude = gain * self.spec.scale
            return amplitude / (amplitude.square() + self.spec.eps).sqrt()
        return gain * (self.spec.scale / (self.spec.scale ** 2 + self.spec.eps) ** .5)


class AccurateGatedBranches(GatedBranches):
    """Identical forward with a cancellation-resistant local RMS VJP."""

    def normalize(self, value):
        return rms_norm(value, self.spec.width, self.spec.eps)


class ResidualTransport(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.spec = spec
        self.gain = (1 + spec.branch_coefficient) ** spec.depth
        self.weight = nn.Parameter(torch.tensor(1 / self.gain))

    def forward(self, inputs):
        state = self.weight * inputs[:, 0]
        for _ in range(self.spec.depth):
            state = (1 + self.spec.branch_coefficient) * state
        return state

    def coefficients(self):
        return (self.weight * self.gain).reshape(1)


def build(spec):
    if isinstance(spec, ContextGates):
        return ContextGatedBranches(spec)
    if isinstance(spec, AccurateBranchGates):
        return AccurateGatedBranches(spec)
    return GatedBranches(spec) if isinstance(spec, BranchGates) else ResidualTransport(spec)
