"""A split as device rows, cut to the longest row of each batch as the historical batches were.

The historical trainer padded each batch to its own longest example
(`experiments/synthetic_trainers/records.py`, `collate`), and a model reads a
batch's width: attention sums over it. A slice of `Rows`, the chunk an
evaluation reads, is therefore cut the same way, and so are training rows
drawn at host indices. A captured update reads the rows at a device index,
which holds no lengths the host could read, at the full width of the split.
A chunk of generated-answer rows also holds what their free generation
reads and is scored on (`generation`).
"""

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
    supervised positions in row-major order, and their `Answers`, or None for prefix rows."""
    rows: torch.Tensor
    supervised: torch.Tensor
    answers: Answers | None


def answers(examples, device):
    """The `Answers` of generated-answer examples, padded with PAD and IGNORE."""
    prompts = tuple(example.prompt for example in examples)
    limits = tuple(example.generation_limit for example in examples)
    widths = tuple(max(len(prompt) + min(step, limit - 1) for prompt, limit in zip(prompts, limits))
                   for step in range(max(limits)))
    context = torch.full((len(examples), widths[-1]), PAD, dtype=torch.long)
    expected = torch.full((len(examples), max(len(example.answer) for example in examples)), IGNORE,
                          dtype=torch.long)
    changes = torch.zeros_like(expected, dtype=torch.bool)
    for row, example in enumerate(examples):
        answer = example.answer
        context[row, :len(example.prompt)] = torch.tensor(example.prompt)
        expected[row, :len(answer)] = torch.tensor(answer)
        changes[row, :len(answer)] = torch.tensor([index == 0 or token != answer[index - 1]
                                                   for index, token in enumerate(answer)])
    lengths = torch.tensor([len(prompt) for prompt in prompts])
    return Answers(prompts, limits, widths, *(tensor.to(device) for tensor in
                                              (context, lengths, torch.tensor(limits), expected, changes)))


class Rows:
    """The examples of a split, padded at the right with PAD, IGNORE targets and no changes.

    Each slice is cut once, on the host, and kept, so the chunks an evaluation
    reads are the same tensors every time it runs: a captured evaluation
    replays on them.
    """

    def __init__(self, examples, device):
        self.examples = examples
        self.lengths = [len(example.tokens) for example in examples]
        self.host = torch.zeros(len(examples), 3, max(self.lengths), dtype=torch.long)
        self.host[:, 0], self.host[:, 1] = PAD, IGNORE
        for row, example in enumerate(examples):
            fields = torch.tensor([example.tokens, example.targets, example.changes])
            self.host[row, :, :fields.shape[1]] = fields
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
            self.chunks[start, stop] = Chunk(self.rows[start:stop, :, :width], supervised.to(self.rows.device),
                                             answers(examples, self.rows.device) if examples[0].prompt else None)
        return self.chunks[start, stop]

    def select(self, indices):
        """The rows at `indices`: cut to the longest of them from host indices, at full width from a device index."""
        if indices.device.type != "cpu":
            return self.rows.index_select(0, indices)
        width = max(self.lengths[index] for index in indices.tolist())
        return self.rows.index_select(0, indices.to(self.rows.device))[:, :, :width]
