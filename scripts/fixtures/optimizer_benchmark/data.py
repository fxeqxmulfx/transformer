"""Frozen character encoding, disjoint splits, and deterministic windows."""

from dataclasses import dataclass
import hashlib
from pathlib import Path
import torch


@dataclass
class TextData:
    train: torch.Tensor
    validation: torch.Tensor
    test: torch.Tensor
    characters: str
    sha256: str
    boundaries: tuple[int, int, int]

    @classmethod
    def load(cls, path, device="cpu"):
        raw = Path(path).read_bytes()
        text = raw.decode("utf-8")
        characters = "".join(sorted(set(text)))
        mapping = {character: i for i, character in enumerate(characters)}
        tokens = torch.tensor([mapping[c] for c in text], dtype=torch.long, device=device)
        first, second = 9 * len(text) // 10, 19 * len(text) // 20
        return cls(tokens[:first], tokens[first:second], tokens[second:], characters,
                   hashlib.sha256(raw).hexdigest(), (first, second, len(text)))


def training_starts(length, context, batch, steps, seed):
    if length <= context or context <= 0 or batch <= 0 or steps <= 0:
        raise ValueError("Training split and budgets must admit next-token windows")
    generator = torch.Generator(device="cpu").manual_seed(10000 + seed)
    return torch.randint(length - context, (steps, batch), generator=generator)


def windows(tokens, starts, context):
    indices = starts[:, None] + torch.arange(context, device=tokens.device)
    return tokens[indices], tokens[indices + 1]


def evaluation_batches(tokens, context, batch):
    starts = torch.arange(0, len(tokens) - context, context, device=tokens.device)
    for offset in range(0, len(starts), batch):
        yield windows(tokens, starts[offset:offset + batch], context)


def select_learning_rates(results, methods):
    """Only final validation loss from screening may select a rate."""
    chosen = {}
    for method in methods:
        candidates = [r for r in results if r["phase"] == "screen" and r["method"] == method
                      and r["status"] == "ok" and math_isfinite(r["validation_loss"])]
        chosen[method] = min(candidates, key=lambda r: (r["validation_loss"], r["lr"]))["lr"] if candidates else None
    return chosen


def math_isfinite(value):
    import math
    return isinstance(value, (float, int)) and math.isfinite(value)
