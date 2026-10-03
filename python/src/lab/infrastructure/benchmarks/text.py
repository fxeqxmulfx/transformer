"""Tiny Shakespeare: next-character prediction on fixed contiguous splits.

Ports of `optimizer_benchmark.data` (the encoding, the splits, the windows)
and of the held-out loss of `full_compile_benchmark.step.FullStep.evaluate`,
which adds each chunk's summed cross entropy into one float32 total.
"""

import hashlib
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from .samplers import WindowSampler

CORPUS = Path(__file__).parent / "data" / "tinyshakespeare.txt"
SHA256 = "53493bf304b639aba6a47400da75c58ce32372b378fd2cd9d16ee987aeb12e4e"


def encode():
    """The corpus as character indices into its sorted character set."""
    raw = CORPUS.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SHA256:
        raise ValueError(f"{CORPUS} is not the Tiny Shakespeare corpus")
    text = raw.decode("utf-8")
    characters = "".join(sorted(set(text)))
    index = {character: position for position, character in enumerate(characters)}
    return torch.tensor([index[character] for character in text], dtype=torch.long), characters


def windows(tokens, starts, window):
    """The rows of `window` characters and their successors at each start."""
    return tokens[starts[:, None] + torch.arange(window + 1, device=tokens.device)]


class TextTask:
    components = ("loss",)

    def __init__(self, spec, data_seed, device):
        tokens, characters = encode()
        first, second = 9 * len(tokens) // 10, 19 * len(tokens) // 20
        self.window, self.device, self.vocab = spec.window, device, len(characters)
        self.train = tokens[:first].to(device)
        self.splits = {name: windows(split, torch.arange(0, len(split) - spec.window, spec.window), spec.window)
                       .to(device) for name, split in (("validation", tokens[first:second]),
                                                       ("test", tokens[second:]))}

    def sampler(self, batch, seed):
        return WindowSampler(len(self.train) - self.window, batch, seed)

    def inputs(self, starts):
        return windows(self.train, starts.to(self.device), self.window)

    def progress(self, seen):
        return {"tokens_seen": seen * self.window}

    def accumulator(self):
        return torch.zeros(1, device=self.device)

    @staticmethod
    def forward(model, rows):
        """Logits at every position, and the next characters."""
        return model(rows[:, :-1]), rows[:, 1:]

    @staticmethod
    def loss(output, targets):
        return F.cross_entropy(output.flatten(0, 1), targets.flatten())

    @staticmethod
    def position_losses(output, targets):
        return TextTask.loss(output, targets).reshape(1)

    @staticmethod
    def accumulate(model, chunk, sums, static):
        """Add a chunk's summed cross entropy to the float32 total, as the historical evaluation did; static either way."""
        output, targets = TextTask.forward(model, chunk)
        sums.add_(F.cross_entropy(output.flatten(0, 1), targets.flatten(), reduction="sum"))

    def metrics(self, sums, examples):
        """The mean loss per character; None when it is not finite, which a stopping policy stops on."""
        loss = sums[0] / (examples * self.window)
        return {"loss": loss if math.isfinite(loss) else None, "examples": examples}

    @staticmethod
    def analyze(history):
        return {}
