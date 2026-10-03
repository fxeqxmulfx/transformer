"""Which updates are observed, sampled and checkpointed.

The rules are those of the historical modular trainer
(`paper_reproduction.grokking.train` with `DiagnosticsConfig`), stated once
for every execution mode.
"""


def canonical(experiment, step):
    """An evaluation recorded in the history: each cadence multiple and the last update."""
    return step % experiment.evaluate.every == 0 or step == experiment.budget.updates


def probe(experiment, step):
    """An evaluation of a neighbor of a canonical observation, recorded apart."""
    every = experiment.evaluate.every
    return experiment.diagnostics.neighbors and step % every in (1, every - 1)


def sampled(experiment, step):
    """An update whose per-tensor norms are measured."""
    every = experiment.diagnostics.every
    return bool(every and step % every == 0) or probe(experiment, step)


def checkpointed(experiment, step):
    """A canonical observation after which the full state is saved."""
    return canonical(experiment, step) and (
        step % experiment.checkpoint.every == 0 or step == experiment.budget.updates)
