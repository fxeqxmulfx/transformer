"""Use cases over the experiments of one file."""

from collections.abc import Callable
from dataclasses import dataclass

from ..domain.experiment import Experiment, differences, require_continuation, study
from ..domain.spec import describe, fingerprint
from .ports import Runs, Trainer


@dataclass(frozen=True)
class Study:
    """The experiments one folder defines; its name is the folder's."""
    name: str
    source: str
    experiments: dict[str, Experiment]

    def __post_init__(self):
        study(self.experiments)

    def select(self, labels):
        """The chosen experiments in file order; no labels selects all."""
        unknown = [label for label in labels if label not in self.experiments]
        if unknown:
            raise KeyError(f"{self.name} defines no {', '.join(unknown)}; "
                           f"it defines {', '.join(self.experiments)}")
        return [(label, experiment) for label, experiment in self.experiments.items()
                if not labels or label in labels]


def survey(study):
    """Each experiment's fingerprint and its differences from the first."""
    first = describe(next(iter(study.experiments.values())))
    return [{"label": label, "fingerprint": fingerprint(experiment),
             "differs_from_first": differences(first, describe(experiment))}
            for label, experiment in study.experiments.items()]


def finished(run, experiment):
    """Whether the run has already reached this experiment's budget."""
    result = run.result()
    return result is not None and result["updates"] == experiment.budget.updates


def require_same_engine(stored, current):
    """A run continues only under the code, framework and device that began it."""
    changed = differences(stored["engine"], current["engine"])
    if changed:
        raise ValueError("The run was trained by another engine; changed: " + ", ".join(changed)
                         + ". Restore that engine, or train the experiment under a new label.")


def run_study(study, labels, runs: Runs, trainer: Trainer, progress: Callable[[str, dict], None]):
    """Train the chosen experiments one after another; a finished run is not trained again.

    Every compatibility check runs before the first update, so a changed
    experiment or engine cannot meet an old run directory halfway through a study.
    """
    sessions = []
    for label, experiment in study.select(labels):
        run = runs.open(label)
        stored = run.description()
        if stored is not None:
            require_continuation(stored, experiment)
            if finished(run, experiment):
                sessions.append((label, experiment, run, None))
                continue
        provenance = trainer.provenance(experiment)
        if stored is not None:
            require_same_engine(run.provenance(), provenance)
        sessions.append((label, experiment, run, provenance))
    results = {}
    for label, experiment, run, provenance in sessions:
        if provenance is None:
            results[label] = run.result()
            continue
        run.begin(describe(experiment), provenance, study.source)
        results[label] = trainer.train(experiment, run, lambda row, label=label: progress(label, row))
    return results
