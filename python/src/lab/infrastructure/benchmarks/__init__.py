"""Benchmark data, sampling, and the losses and metrics a trainer needs from it.

A task is built for one device and holds there the rows of the splits it
evaluates (`splits`). It draws training batches with `sampler(batch, seed)`,
whose indices `inputs` turns into a batch; `forward` maps a batch to the
supervised logits and their targets, `loss` to the training loss, and
`position_losses` to the mean loss at each supervised position, named by
`components`. Exhaustive evaluation adds each chunk of a split's rows into
`accumulator()` with `accumulate`, and `metrics` reads the totals.
`progress` states how much training data a number of examples is, and
`analyze` summarizes the history of observations.
"""

from ...domain import benchmarks
from .modular import ModularTask
from .text import TextTask

TASKS = {benchmarks.ModularDivision: ModularTask, benchmarks.TinyShakespeare: TextTask}


def build_task(spec, data_seed, device):
    if type(spec) not in TASKS:
        raise NotImplementedError(f"No task for {spec!r}")
    return TASKS[type(spec)](spec, data_seed, device)
