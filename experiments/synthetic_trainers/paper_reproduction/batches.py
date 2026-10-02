"""Explicit epoch-tail control for the Section 4 modular-division follow-up."""

import torch


def next_batch(permutation, cursor, generator, batch_size, policy="short_final"):
    """Keep shuffled epochs, either ending at their tail or filling across it.

    The default exactly retains the previous short last batch. The wrap control
    consumes the same shuffled example stream but has more examples at fixed
    update count. It is a separately labeled sampling adaptation.
    """
    if policy not in ("short_final", "wrap_epoch") or batch_size < 1:
        raise ValueError("Unknown batch policy or nonpositive batch size")
    size = len(permutation)
    if size < 1 or not 0 <= cursor <= size:
        raise ValueError("Invalid shuffled epoch or cursor")
    if cursor == size:
        permutation = torch.randperm(size, generator=generator)
        cursor = 0
    start, wraps, remaining, parts = cursor, 0, batch_size, []
    while remaining:
        count = min(remaining, size - cursor)
        parts.append(permutation[cursor:cursor + count])
        cursor += count
        remaining -= count
        if policy == "short_final" or not remaining:
            break
        permutation = torch.randperm(size, generator=generator)
        cursor = 0
        wraps += 1
    return torch.cat(parts), permutation, cursor, {
        "cursor_before": start, "cursor_after": cursor, "epoch_tail": cursor == size,
        "epoch_wraps_in_batch": wraps,
    }
