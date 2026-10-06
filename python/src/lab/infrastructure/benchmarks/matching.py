"""Numerical answer-only check of the Lean order-sensitive physical head.

Source: matchingOrderTokens, matchingOrderTarget and
matchingOrderError_uniform_lower at 5d91bc4. No route labels are supplied.
All splits refer to the same two formal observations. Loss is a sum, not
mean, so its scale agrees exactly with the Lean error 1/8.
"""

import torch

from ..nn.sparsemax import causal_sparsemax
from .samplers import EpochSampler


class MatchingTask:
    components = ("loss",)
    vocab = 2

    def __init__(self, spec, data_seed, device):
        self.spec, self.device = spec, device
        self.tokens = torch.tensor([[0, 1], [1, 0]], device=device)
        self.targets = torch.tensor(spec.targets, dtype=torch.float64, device=device).reshape(2, 1, 1)
        rows = torch.arange(2, device=device)
        self.splits = {"train": rows, "validation": rows}

    def sampler(self, batch, seed):
        return EpochSampler(2, batch, False, seed)

    def inputs(self, indices, static=False):
        return indices.to(self.device)

    def forward(self, model, batch, supervised=False):
        rows = torch.ones(len(batch), 1, dtype=torch.long, device=self.device)
        return model(self.tokens[batch], rows), self.targets[batch]

    @staticmethod
    def loss(output, targets):
        return (output - targets).square().sum()

    @staticmethod
    def position_losses(output, targets):
        return MatchingTask.loss(output, targets).reshape(1)

    def accumulator(self):
        return torch.zeros(1, dtype=torch.float64, device=self.device)

    def accumulate(self, model, chunk, sums, static):
        sums[0].add_(self.loss(*self.forward(model, chunk)))

    def metrics(self, sums, rows):
        return {"loss": sums[0]}

    @staticmethod
    def progress(seen):
        return {"examples_seen": seen, "epochs_seen": seen / 2}

    def inspect(self, model, batch):
        with torch.no_grad():
            prediction, targets = self.forward(model, self.splits["train"])
            count = int(model.active)
            supports = []
            for i in range(count):
                q, k = model.q[i][self.tokens], model.k[i][self.tokens]
                supports.append((causal_sparsemax(q @ k.transpose(-1, -2))[:, 1] > 0).tolist())
            return {"atomic": {**model.inspect(), "predictions": prediction.flatten().tolist(),
                               "targets": targets.flatten().tolist(), "head_supports": supports,
                               "q": model.q[:count].tolist(), "k": model.k[:count].tolist(),
                               "values": model.values[:count].tolist()}}

    def observe(self, model, batch):
        return self.inspect(model, batch)

    @staticmethod
    def analyze(history):
        return {"generalization_test": False}
