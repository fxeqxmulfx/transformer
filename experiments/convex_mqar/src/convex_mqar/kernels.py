"""Optional loss compilation without wrapping the saved model or changing SGD."""

from contextlib import nullcontext

import torch
from torch.nn import functional as F


def autocast(config):
    if config.device == "cuda" and config.precision == "bf16":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def loss_for(model, config, compiled=False):
    def objective(tokens, positions, labels):
        with autocast(config):
            logits = model(tokens, positions)
            return F.cross_entropy(logits.flatten(0, 1), labels.flatten())

    # Optimizer, learning-rate schedule, and batch boundaries stay outside
    # the compiled graph. Checkpoints retain the ordinary model state dict.
    return torch.compile(objective, fullgraph=True, dynamic=False) if compiled else objective
