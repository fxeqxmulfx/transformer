"""Batches drawn from shuffled epochs of the training split.

A port of `paper_reproduction.batches.next_batch`: one generator draws every
permutation, in the same order, so a run walks the historical batch sequence.
With tail "short" an epoch ends with its remainder as a smaller batch; with
"wrap" a batch fills up from the next permutation.
"""

import torch


class EpochSampler:
    def __init__(self, size, budget, seed):
        self.batch, self.wrap = budget.batch, budget.tail == "wrap"
        self.generator = torch.Generator().manual_seed(seed)
        self.permutation = torch.randperm(size, generator=self.generator)
        self.cursor = 0

    def next(self):
        """The next batch as (permutation, start, count) slices, and its place in the epochs."""
        size = len(self.permutation)
        if self.cursor == size:
            self.permutation = torch.randperm(size, generator=self.generator)
            self.cursor = 0
        start, wraps, remaining, parts = self.cursor, 0, self.batch, []
        while remaining:
            count = min(remaining, size - self.cursor)
            parts.append((self.permutation, self.cursor, count))
            self.cursor += count
            remaining -= count
            if not self.wrap or not remaining:
                break
            self.permutation = torch.randperm(size, generator=self.generator)
            self.cursor = 0
            wraps += 1
        return parts, {"cursor_before": start, "cursor_after": self.cursor, "epoch_tail": self.cursor == size,
                       "epoch_wraps_in_batch": wraps}

    def state(self):
        return {"batch_generator_state": self.generator.get_state(), "permutation": self.permutation,
                "cursor": self.cursor}

    def restore(self, checkpoint):
        self.generator.set_state(checkpoint["batch_generator_state"].cpu())
        self.permutation, self.cursor = checkpoint["permutation"].cpu(), checkpoint["cursor"]


def gather(parts):
    """A batch's training-row indices, on the host."""
    return torch.cat([permutation[start:start + count] for permutation, start, count in parts])
