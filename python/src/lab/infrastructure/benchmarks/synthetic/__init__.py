"""Synthetic benchmarks: the historical algorithmic suite, sampled exactly as it was.

Each task of `lab.domain.tasks` and `lab.domain.generative` has a generator
here, of the same name, that samples it at given problem lengths and checks
every example against an oracle on its raw input (`base`). Splits draw from
seeds derived from the task and the data seed, and keep the historical
fingerprints (`splits`); a study corrupts training labels (`noise`).
"""

from ....domain import generative, tasks
from . import arithmetic, bits, control, crasp, prefixes, retrieval, sequences

GENERATORS = {
    tasks.MQAR: retrieval.MQAR, tasks.Lookup: retrieval.Lookup, tasks.Dyck: prefixes.Dyck,
    tasks.AlternatingBlocks: prefixes.AlternatingBlocks, tasks.TypedDyck: prefixes.TypedDyck,
    tasks.CRASP: crasp.CRASP, generative.Histogram: sequences.Histogram,
    generative.DoubleHistogram: sequences.DoubleHistogram, generative.Mode: sequences.Mode,
    generative.MostFrequent: sequences.MostFrequent, generative.Copy: sequences.Copy,
    generative.Reverse: sequences.Reverse, generative.Sort: sequences.Sort, generative.Count: sequences.Count,
    generative.Addition: arithmetic.Addition, generative.Parity: bits.Parity,
    generative.BooleanAnd: bits.BooleanAnd, generative.RandomLM: control.RandomLM,
}


def generator(problem):
    """The generator of a `Problem`: a task at its problem lengths."""
    if type(problem.task) not in GENERATORS:
        raise NotImplementedError(f"No generator for {problem.task!r}")
    return GENERATORS[type(problem.task)](problem)
