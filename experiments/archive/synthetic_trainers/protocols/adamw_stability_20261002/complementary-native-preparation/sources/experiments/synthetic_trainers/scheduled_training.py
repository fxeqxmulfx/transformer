"""Prepare an explicit fixed-rate/annealed-rate native AdamW calibration.

Convexifying Transformers, Section 4 supplies the modular task. This is a
separately labeled scheduling adaptation, not a reproduction of its schedule
or a guarantee of stable training. No target observation changes the schedule.
"""

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
from unittest.mock import patch

from .paper_reproduction import grokking
from .paper_reproduction.provenance import ROOT, source_hashes
from .scheduled_rates import expected_rate, validate_schedule


@dataclass(frozen=True)
class ScheduledRunConfig(grokking.RunConfig):
    steps: int = 300000
    learning_rate_schedule: str = "constant"
    anneal_start: int = 150000
    anneal_end: int = 250000
    final_rate_factor: float = .1

    def __post_init__(self):
        super().__post_init__()
        validate_schedule(asdict(self))


def learning_rate(config, completed_steps):
    if type(config) is not ScheduledRunConfig:
        raise TypeError("Scheduled rates require an explicitly tagged config")
    return expected_rate(vars(config), completed_steps)


def training_sources():
    result = source_hashes()
    for name in ("scheduled_training.py", "scheduled_rates.py"):
        path = Path(__file__).parent / name
        result[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return dict(sorted(result.items()))


def train(config, directory, *, resume=False, **kwargs):
    """Preserve the original loop, native optimizer, sampling and diagnostics.

    Record the schedule and source before resume validation; all diagnostic
    rates come from the actual optimizer groups used by the original loop.
    The scientific cap remains fixed. Small CPU budget extensions are fixtures.
    """
    if type(config) is not ScheduledRunConfig:
        raise TypeError("Schedule training requires an explicit ScheduledRunConfig")
    with patch.object(grokking, "learning_rate", learning_rate), patch.object(grokking, "source_hashes", training_sources):
        return grokking.train(config, directory, resume=resume, **kwargs)
