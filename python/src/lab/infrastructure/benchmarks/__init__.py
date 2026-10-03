"""Benchmark data and the losses and metrics a trainer needs from it.

A task holds the token rows of each split (`splits`, training first), the
vocabulary size, and the operations a trainer applies to a batch: `forward`
to the supervised logits and their targets, `loss`, `position_losses`, and
`accumulate`/`metrics` for exhaustive evaluation.
"""

from ...domain import benchmarks
from .modular import ModularTask

TASKS = {benchmarks.ModularDivision: ModularTask}


def build_task(spec, data_seed):
    if type(spec) not in TASKS:
        raise NotImplementedError(f"No task for {spec!r}")
    return TASKS[type(spec)](spec, data_seed)
