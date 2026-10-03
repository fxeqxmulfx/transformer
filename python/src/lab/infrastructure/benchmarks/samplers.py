"""Training batches as slices of index tensors, drawn on the host.

`next()` returns a batch as (indices, start, count) slices and its place in
the sampling order; `gather` joins the slices. A sampler draws its indices
in tensors that recur rarely, such as an epoch's permutation, so a graph
stepper uploads each tensor once and copies batches out of it on the device.
`sizes()` are the batch sizes that recur, and `traced` the keys of a batch's
place that the gradient trace records.
"""

import torch


class EpochSampler:
    """Batches drawn from shuffled epochs of `size` training rows.

    A port of `paper_reproduction.batches.next_batch`: one generator draws every
    permutation, in the same order, so a run walks the historical batch
    sequence. Without `wrap` an epoch ends with its remainder as a smaller
    batch; with it a batch fills up from the next permutation.
    """
    traced = ("epoch_tail",)

    def __init__(self, size, batch, wrap, seed):
        self.batch, self.wrap = batch, wrap
        self.generator = torch.Generator().manual_seed(seed)
        self.permutation = torch.randperm(size, generator=self.generator)
        self.cursor = 0

    def sizes(self):
        """The full batch and, with short tails, an epoch's remainder."""
        size = len(self.permutation)
        if self.wrap:
            return [self.batch]
        return sorted({min(self.batch, size), size % self.batch} - {0})

    def next(self):
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


class WindowSampler:
    """`batch` uniform window starts below `high` per update, `block` updates at a time.

    A port of `optimizer_benchmark.data.training_starts`, which drew the starts
    of every update at once, `torch.randint(high, (updates, batch))`; the CPU
    generator yields the same numbers in blocks.
    """
    traced = ()

    def __init__(self, high, batch, seed, block=1024):
        self.high, self.batch, self.block = high, batch, block
        self.generator = torch.Generator().manual_seed(seed)
        self.starts, self.cursor = self.draw(), 0

    def draw(self):
        return torch.randint(self.high, (self.block * self.batch,), generator=self.generator)

    def sizes(self):
        return [self.batch]

    def next(self):
        if self.cursor == self.block:
            self.starts, self.cursor = self.draw(), 0
        self.cursor += 1
        return [(self.starts, (self.cursor - 1) * self.batch, self.batch)], {}

    def state(self):
        return {"window_generator_state": self.generator.get_state(), "starts": self.starts, "cursor": self.cursor}

    def restore(self, checkpoint):
        self.generator.set_state(checkpoint["window_generator_state"].cpu())
        self.starts, self.cursor = checkpoint["starts"].cpu(), checkpoint["cursor"]


def gather(parts):
    """A batch's indices, on the host."""
    return torch.cat([indices[start:start + count] for indices, start, count in parts])
