"""What the use cases need from the outside world."""

from collections.abc import Callable
from typing import Any, Protocol

from ..domain.experiment import Experiment


class Run(Protocol):
    """The record of one experiment's training, across sessions."""

    def description(self) -> dict | None:
        """The stored description of the experiment, if the run has begun."""

    def provenance(self) -> dict | None:
        """The provenance of the latest session."""

    def result(self) -> dict | None:
        """The summary of the latest session that reached its budget."""

    def begin(self, description: dict, provenance: dict, source: str) -> None:
        """Open a session: record the experiment, the provenance and the defining file."""

    # What a trainer reads and writes during a session.

    def checkpoint(self, device: Any) -> dict | None:
        """The last saved state, if any."""

    def save(self, checkpoint: dict) -> None:
        """Replace the saved state."""

    def records(self, stream: str) -> list[dict]:
        """The rows of a record stream: history, probes, diagnostics or gradients."""

    def record(self, stream: str, row: dict) -> None:
        """Append a row to a record stream."""

    def rewind(self, step: int) -> None:
        """Drop every record of an update after `step`."""

    def finish(self, result: dict) -> None:
        """Record the summary of a session that reached its budget."""


class Runs(Protocol):
    def open(self, study: str, label: str) -> Run:
        """The run of one labeled experiment of a study."""


class Trainer(Protocol):
    def provenance(self, experiment: Experiment) -> dict:
        """What determines the trajectory besides the experiment; its `engine` entry must
        match for a run to continue."""

    def train(self, experiment: Experiment, run: Run, progress: Callable[[dict], None]) -> dict:
        """Train from the run's last checkpoint, or from scratch, to the budget."""
