"""What the use cases need from the outside world."""

from collections.abc import Callable
from typing import Protocol

from ..domain.experiment import Experiment


class Run(Protocol):
    """The record of one experiment's training run."""

    def description(self) -> dict | None:
        """The stored description of the experiment, if the run has begun."""

    def begin(self, description: dict, source: str) -> None:
        """Record the experiment and the text of the file that defined it."""


class Runs(Protocol):
    def open(self, study: str, label: str) -> Run:
        """The run of one labeled experiment of a study."""


class Trainer(Protocol):
    def train(self, experiment: Experiment, run: Run, progress: Callable[[dict], None]) -> dict:
        """Train from the run's last checkpoint, or from scratch, to the budget."""
