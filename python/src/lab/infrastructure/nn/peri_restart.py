"""Pinned-weight interventions; the actual PeriRotate90 forward is reused.

Source: domain.peri_restart.PeriRotate90Restart. Checkpoints contain training
query weights y. No optimizer, sampler or averaging history is imported.
"""

import hashlib
from pathlib import Path

import torch

from ..rotary_geometry import multiplicative_ordering
from .peri_rotary import PeriRotate90


def alias_projection(embedding, size):
    """Average multiplicative cosets, preserving the zero row exactly.

    Each orbit mean is calculated once and copied to its members. This avoids
    summation-order differences giving a nominally symmetric FP32 start tiny
    initial asymmetries. It is a diagnostic projection, never a training update.
    """
    prime = len(embedding)
    if (prime - 1) % size:
        raise ValueError("Alias size must divide the multiplicative group order")
    generator, _ = multiplicative_ordering(prime)
    subgroup = [pow(generator, (prime - 1) // size * j, prime) for j in range(size)]
    result = embedding.detach().clone()
    remaining = set(range(1, prime))
    while remaining:
        representative = min(remaining)
        orbit = sorted({representative * h % prime for h in subgroup})
        result[orbit] = embedding[orbit].mean(0)
        remaining.difference_update(orbit)
    return result


def restart(spec, vocab, seed):
    """Load verified source weights, intervene, then select trainable tensors."""
    path = Path(spec.checkpoint)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[5] / path
    if hashlib.sha256(path.read_bytes()).hexdigest() != spec.sha256:
        raise ValueError("Restart checkpoint SHA256 mismatch")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    groups = checkpoint["optimizer"]["param_groups"]
    if len(groups) != 1 or groups[0].get("training") is not True:
        raise ValueError("Restart requires a Schedule-Free training-query checkpoint")
    model = PeriRotate90(spec.network, vocab, seed)
    model.load_state_dict(checkpoint["model"], strict=True)
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        if spec.alias_size > 1:
            model.E.copy_(alias_projection(model.E, spec.alias_size))
        if spec.noise:
            for parameter in model.parameters():
                noise = torch.randn(parameter.shape, generator=generator, dtype=parameter.dtype)
                parameter.add_(noise * (spec.noise * parameter.square().mean().sqrt()))
    model.E.requires_grad_(spec.train_embedding)
    for parameter in model.hidden_matrices():
        parameter.requires_grad_(spec.train_branch)
    return model
