"""A split as device rows, cut to the longest row of each batch as the historical batches were.

The historical trainer padded each batch to its own longest example
(`experiments/synthetic_trainers/records.py`, `collate`), and a model reads a
batch's width: attention sums over it. A slice of `Rows`, the chunk an
evaluation reads, is therefore cut the same way, and so are training rows
drawn at host indices. A captured update reads the rows at a device index,
which holds no lengths the host could read, at the full width of the split.
"""

from typing import NamedTuple

import torch

from .vocabulary import IGNORE, PAD


class Chunk(NamedTuple):
    """Consecutive rows (tokens, targets, changes on the second axis), and the flat indices of their
    supervised positions in row-major order."""
    rows: torch.Tensor
    supervised: torch.Tensor


class Rows:
    """The examples of a split, padded at the right with PAD, IGNORE targets and no changes.

    Each slice is cut once, on the host, and kept, so the chunks an evaluation
    reads are the same tensors every time it runs: a captured evaluation
    replays on them.
    """

    def __init__(self, examples, device):
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
            self.chunks[start, stop] = Chunk(self.rows[start:stop, :, :width], supervised.to(self.rows.device))
        return self.chunks[start, stop]

    def select(self, indices):
        """The rows at `indices`: cut to the longest of them from host indices, at full width from a device index."""
        if indices.device.type != "cpu":
            return self.rows.index_select(0, indices)
        width = max(self.lengths[index] for index in indices.tolist())
        return self.rows.index_select(0, indices.to(self.rows.device))[:, :, :width]
