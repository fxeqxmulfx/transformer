"""Benchmark data, sampling, and the losses and metrics a trainer needs from it.

A task is built for one device and holds there the rows of the splits it
evaluates (`splits`). It draws training batches with `sampler(batch, seed)`,
whose indices `inputs` turns into a batch: host indices for an update issued
eagerly, or the device index that a captured update reads without
synchronizing. A `static` batch has the shape its size fixes, whatever rows
it holds, as a captured or compiled update needs. `forward` maps a batch to
the supervised logits and their targets, reading out, when `supervised`, no
more positions than a training row is supervised at; `loss` maps them to the
training loss, and `position_losses` to the mean loss at each supervised
position, named by `components`. Exhaustive evaluation adds each chunk of a
split's rows, `splits[name][start:stop]`, into `accumulator()` with
`accumulate`, and `metrics` reads the totals of the split's rows. A `static`
evaluation fixes all its shapes in advance and never waits on the device, so
a captured evaluation can replay it. Two splits may share their rows, and a
split without rows is None: its metrics are None. `progress` states how much
training data a number of examples is; `observe` measures a model at an
observation beyond its splits' metrics, and `inspect` measures the last and
the best model; `analyze` summarizes the history of observations.
"""

from ...domain.benchmarks import ModularDivision, TinyShakespeare
from ...domain.synthetic import Synthetic
from .modular import ModularTask
from .synthetic.training import SyntheticTask
from .text import TextTask

TASKS = {ModularDivision: ModularTask, TinyShakespeare: TextTask, Synthetic: SyntheticTask}


def build_task(spec, data_seed, device):
    if type(spec) not in TASKS:
        raise NotImplementedError(f"No task for {spec!r}")
    return TASKS[type(spec)](spec, data_seed, device)
