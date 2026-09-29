"""Small deterministic fixtures with an independent raw-sequence recall oracle."""

from dataclasses import replace

import numpy as np
import torch

from convex_mqar.certify import make_example
from convex_mqar.config import Config


def config(**overrides):
    small = replace(Config(), vocab=16, lengths=(8,), widths=(8,),
                    learning_rates=(0.003, 0.01), train_examples=67,
                    validation_examples=65, test_examples=19, epochs=3,
                    device="cpu", precision="fp32")
    return replace(small, **overrides)


def batch(seed=0, count=7, length=16, vocab=64):
    rng = np.random.default_rng(seed)
    rows = [make_example(rng, length, length // 4, vocab, 0.1)
            for _ in range(count)]
    return tuple(torch.from_numpy(np.stack(items)).long() for items in zip(*rows))


def recall_from_raw(tokens, position):
    """Find the last previous occurrence and return its causal successor."""
    matches = [j for j in range(position - 1) if tokens[j] == tokens[position]]
    return tokens[matches[-1] + 1] if matches else -1
