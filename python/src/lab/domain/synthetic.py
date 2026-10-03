"""Synthetic benchmarks: an algorithmic task sampled into finite splits and tested at longer lengths."""

from dataclasses import dataclass
import math

from .benchmarks import Benchmark
from .generative import RandomLM
from .spec import Spec, require, require_kind
from .tasks import Problem, Task


@dataclass(frozen=True)
class Memorization(Spec):
    """The finite pools of a memorization study, and the label noise of its training pool.

    Source: the study profiles of the historical synthetic suite
    (`experiments/synthetic_trainers/studies.py`, `corpus.py`, `sweeps.py`).
    With `disjoint` no input recurs within or across the train, validation
    and test pools; otherwise each is sampled on its own, with replacement.
    `noise` replaces each training label that has another legal value, with
    that probability, by one of them drawn uniformly, once per input and
    position from `noise_seed` (arXiv:1912.02292v1, Section 4); BOS, EOS,
    separators and index hints keep their labels. A `pool` larger than the
    training split is sampled in full and training takes its first rows, so
    runs on nested training pools share their validation and test pools.
    """
    disjoint: bool = False
    noise: float = 0.0
    noise_seed: int = 0
    pool: int | None = None

    def check(self):
        require(math.isfinite(self.noise) and 0 <= self.noise <= 1, "Label noise is a probability")
        require(self.noise_seed >= 0, "The noise seed is nonnegative")
        require(self.pool is None or self.pool >= 1, "A pool holds at least one input")


@dataclass(frozen=True)
class Synthetic(Benchmark):
    """An algorithmic task, trained at problem lengths `min_length` to `length` and tested longer.

    Source: the historical synthetic suite (`experiments/synthetic_trainers`,
    `training.train_run`). The `train`, `validation` and `test` splits are
    sampled from the task at the training lengths, each from a seed of its
    own derived from the data seed and the task. Each `ood` length L adds
    the held-out distribution `length-L` of problems of length L, and the
    task may add its own (`Task.transfers`). Without `min_length` every
    problem has length `length`, as with `min_length=length`, but the splits
    are seeded differently, as they were.
    """
    task: Task
    length: int
    min_length: int | None = None
    train: int = 512
    validation: int = 128
    test: int = 128
    ood: tuple[int, ...] = ()
    study: Memorization | None = None

    def check(self):
        require_kind(self.task, Task, "task")
        require(1 <= self.problem.minimum <= self.length, "Lengths must satisfy 1 <= min_length <= length")
        require(min(self.train, self.validation, self.test) >= 1, "Every split holds an example")
        require(len(set(self.ood)) == len(self.ood) and all(length > self.length for length in self.ood),
                "OOD lengths are distinct and exceed the training maximum")
        for problem in (self.problem, *self.probes.values()):
            problem.task.check_lengths(problem.minimum, problem.length)
        if self.study is not None:
            require_kind(self.study, Memorization, "study")
            require(self.study.pool is None or self.study.pool >= self.train, "The pool holds the training split")
            require(not isinstance(self.task, RandomLM) or not (self.study.noise or self.study.disjoint),
                    "The random control's answers are random already, and sampled with replacement")

    @property
    def problem(self):
        """The training distribution, which the validation and test splits share."""
        return Problem(self.task, self.length, self.min_length)

    @property
    def probes(self):
        """The held-out distributions by name: `length-L` for each OOD length, then the task's own."""
        return {**{f"length-{length}": Problem(self.task, length, None) for length in self.ood},
                **self.task.transfers(self.length, self.min_length, self.ood)}

    @property
    def context(self):
        return max(problem.task.context(problem.length) for problem in (self.problem, *self.probes.values()))
