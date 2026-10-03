"""Use cases over the experiments of one file."""

from collections.abc import Callable
from dataclasses import dataclass

from ..domain.experiment import Experiment, differences, require_continuation, study
from ..domain.spec import describe, fingerprint
from .ports import Runs, Trainer


@dataclass(frozen=True)
class Study:
    """The experiments one file defines; its name is the file's stem."""
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


def run_study(study, labels, runs: Runs, trainer: Trainer, progress: Callable[[str, dict], None]):
    """Train the chosen experiments one after another.

    Every compatibility check runs before the first update, so a changed
    experiment cannot meet an old run directory halfway through a study.
    """
    chosen = study.select(labels)
    opened = []
    for label, experiment in chosen:
        run = runs.open(study.name, label)
        stored = run.description()
        if stored is not None:
            require_continuation(stored, experiment)
        opened.append((label, experiment, run))
    results = {}
    for label, experiment, run in opened:
        run.begin(describe(experiment), study.source)
        results[label] = trainer.train(experiment, run, lambda row, label=label: progress(label, row))
    return results
