"""A split as device rows, cut to the longest row of each batch as the historical batches were.

The historical trainer padded each batch to its own longest example
(`experiments/synthetic_trainers/records.py`, `collate`), and a model reads a
batch's width: attention sums over it. A slice of `Rows`, the chunk an
evaluation reads, is therefore cut the same way, and so are training rows
drawn for an update issued eagerly. A static update, captured or compiled,
reads them at the full width of the split, so that its shape depends on its
size alone.
A chunk of generated-answer rows also holds what their free generation
reads and is scored on (`generation`).
"""

from array import array
from itertools import chain, repeat
from typing import NamedTuple

import torch

from .vocabulary import IGNORE, PAD


class Answers(NamedTuple):
    """The prompts and generation limits of a chunk's rows, on the host and the device, and their answers.

    `widths[s]` is the widest context a row can hold at step s of a
    generation; `context` holds the prompts at the widest of them. An
    answer changes where its token differs from the one before it.
    """
    prompts: tuple[tuple[int, ...], ...]
    limits: tuple[int, ...]
    widths: tuple[int, ...]
    context: torch.Tensor
    lengths: torch.Tensor
    bounds: torch.Tensor
    answers: torch.Tensor
    changes: torch.Tensor


class Chunk(NamedTuple):
    """Consecutive rows (tokens, targets, changes on the second axis), the flat indices of their
    supervised positions in row-major order, and their `Answers`, or None for rows not generated."""
    rows: torch.Tensor
    supervised: torch.Tensor
    answers: Answers | None


def answers(prompts, limits, expected, device):
    """The `Answers` of prompts generating at most `limits` tokens, scored on `expected`, padded with PAD and IGNORE."""
    prompts, limits = tuple(prompts), tuple(limits)
    widths = tuple(max(len(prompt) + min(step, limit - 1) for prompt, limit in zip(prompts, limits))
                   for step in range(max(limits)))
    context = torch.full((len(prompts), widths[-1]), PAD, dtype=torch.long)
    reference = torch.full((len(prompts), max(map(len, expected))), IGNORE, dtype=torch.long)
    changes = torch.zeros_like(reference, dtype=torch.bool)
    for row, (prompt, answer) in enumerate(zip(prompts, expected, strict=True)):
        context[row, :len(prompt)] = torch.tensor(prompt)
        reference[row, :len(answer)] = torch.tensor(answer)
        changes[row, :len(answer)] = torch.tensor([index == 0 or token != answer[index - 1]
                                                   for index, token in enumerate(answer)])
    lengths = torch.tensor([len(prompt) for prompt in prompts])
    return Answers(prompts, limits, widths, *(tensor.to(device) for tensor in
                                              (context, lengths, torch.tensor(limits), reference, changes)))


class Rows:
    """The examples of a split, padded at the right with PAD, IGNORE targets and no changes.

    Each slice is cut once, on the host, and kept, so the chunks an evaluation
    reads are the same tensors every time it runs: a captured evaluation
    replays on them. Without `generate`, generated answers are scored by
    teacher forcing alone, as the historical trainer scored a training split.
    """

    def __init__(self, examples, device, generate=True):
        self.examples = examples
        self.generates = generate and bool(examples[0].prompt)
        self.lengths = [len(example.tokens) for example in examples]
        width = max(self.lengths)

        def padded(example):
            pad = width - len(example.tokens)
            return chain(example.tokens, repeat(PAD, pad), example.targets, repeat(IGNORE, pad),
                         example.changes, repeat(0, pad))

        # One buffer of 64-bit integers, filled without a tensor per example.
        values = array("q", chain.from_iterable(map(padded, examples)))
        self.host = torch.frombuffer(values, dtype=torch.long).view(len(examples), 3, width)
        self.rows = self.host.to(device)
        self.chunks = {}

    def __len__(self):
        return len(self.lengths)

    def __getitem__(self, part):
        """The chunk of the rows in the slice `part`, cut to its longest row."""
        start, stop, step = part.indices(len(self))
        if step != 1 or start >= stop:
            raise IndexError("A chunk is a nonempty run of consecutive rows")
        if (start, stop) not in self.chunks:
            width = max(self.lengths[start:stop])
            supervised = (self.host[start:stop, 1, :width] != IGNORE).flatten().nonzero().flatten()
            examples = self.examples[start:stop]
            generated = None if not self.generates else answers(
                [example.prompt for example in examples], [example.generation_limit for example in examples],
                [example.answer for example in examples], self.rows.device)
            self.chunks[start, stop] = Chunk(self.rows[start:stop, :, :width], supervised.to(self.rows.device),
                                             generated)
        return self.chunks[start, stop]

    def select(self, indices, static=False):
        """The rows at `indices`: at the full width when `static`, otherwise cut to the longest of them."""
        if static:
            return self.rows.index_select(0, indices.to(self.rows.device))
        width = max(self.lengths[index] for index in indices.tolist())
        return self.rows.index_select(0, indices.to(self.rows.device))[:, :, :width]
