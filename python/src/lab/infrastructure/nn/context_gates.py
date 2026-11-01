"""Task158 conditional amplitudes; see experiments/context_gains, Eqs. (1)-(3).

The two-dimensional supplied feature replaces a full transformer branch input.
Both placements use task155's accurate finite-epsilon RMS backward; no native
backward cancellation is being rediscovered. The production proposal uses a
zero-initialized linear row on the already normalized full branch input.
"""

import torch
from torch import nn

from .rms_backward import rms_norm


class ContextGatedBranches(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.spec = spec
        self.weight = nn.Parameter(torch.zeros(spec.depth, 2))

    def forward(self, inputs):
        sign = inputs[:, 0, 0]
        features = torch.stack((torch.ones_like(sign), sign), -1)
        gains = (2 * (features @ self.weight.T).sigmoid())[:, :, None]
        branches = self.spec.scale * inputs[:, 1:]
        if self.spec.placement == "before":
            branches = rms_norm(gains * branches, self.spec.width, self.spec.eps)
        else:
            branches = gains * rms_norm(branches, self.spec.width, self.spec.eps)
        return inputs[:, 0] + branches.sum(1)

    def coefficients(self):
        logits = torch.stack((self.weight[:, 0] - self.weight[:, 1],
                              self.weight[:, 0] + self.weight[:, 1]))
        gains = 2 * logits.sigmoid()
        if self.spec.placement == "before":
            amplitude = gains * self.spec.scale
            return amplitude / (amplitude.square() + self.spec.eps).sqrt()
        return gains * self.spec.scale / (self.spec.scale**2 + self.spec.eps)**.5
