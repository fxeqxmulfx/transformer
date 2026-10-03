"""Synthetic benchmarks: an algorithmic task sampled into finite splits and tested at longer lengths."""

from dataclasses import dataclass

from .benchmarks import Benchmark
from .generative import RandomLM
from .memorization import Memorization
from .spec import require, require_kind
from .tasks import Problem, Task

METRICS = ("token_accuracy", "sequence_accuracy", "balanced_accuracy", "final_answer_accuracy")


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

    A generative task is scored on free generation (`evaluate` of
    `experiments/synthetic_trainers/metrics.py`): each answer is generated
    greedily from its prompt alone, until EOS or a limit set by the prompt,
    and the accuracies score what was generated; the loss stays that of
    teacher forcing, whose own metrics are reported as `teacher_forced`. Its
    final answer accuracy scores the final answer alone (`Task.final_token`).

    Validation is observed; the best observation has the highest sequence
    accuracy, then balanced accuracy, then the lowest loss, after the final
    answer accuracy when that is the target `metric` (`validation_rank` of
    the same module), and the test splits,
    `test` and `test/<name>` of each held-out distribution, are evaluated
    once, on its model. The run reports the first observation whose
    validation `metric` reaches `target`.

    A `study` (`Memorization`) also observes the training split by teacher
    forcing alone, with its observed labels (`train`) and the oracle's
    (`train_clean`), each held-out distribution at the size of validation
    (`validation/<name>`), and the rows of validation and of each held-out
    split whose input no training row has (`<split>/novel`), None when
    there are none (`studies.py`). It evaluates the test splits on the last
    model too, before the best is restored.
    """
    task: Task
    length: int
    min_length: int | None = None
    train: int = 512
    validation: int = 128
    test: int = 128
    ood: tuple[int, ...] = ()
    study: Memorization | None = None
    target: float | None = 0.95
    metric: str = "sequence_accuracy"

    def check(self):
        require_kind(self.task, Task, "task")
        require(1 <= self.problem.minimum <= self.length, "Lengths must satisfy 1 <= min_length <= length")
        require(min(self.train, self.validation, self.test) >= 1, "Every split holds an example")
        require(len(set(self.ood)) == len(self.ood) and all(length > self.length for length in self.ood),
                "OOD lengths are distinct and exceed the training maximum")
        require(self.target is None or 0 <= self.target <= 1, "The target is an accuracy, or None")
        require(self.metric in METRICS, f"The target metric is one of {', '.join(METRICS)}")
        require(self.metric != "final_answer_accuracy" or self.task.generative,
                "The final answer accuracy scores generated answers only")
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

    @property
    def observed(self):
        if self.study is None:
            return ("validation",)
        return ("validation", "validation/novel", "train", "train_clean",
                *(f"validation/{name}{novel}" for name in self.probes for novel in ("", "/novel")))

    @property
    def selection(self):
        return "validation"

    @property
    def final(self):
        return ("test", *(f"test/{name}" for name in self.probes))

    @property
    def last(self):
        return () if self.study is None else self.final

    def rank(self, metrics):
        rank = metrics["sequence_accuracy"], metrics["balanced_accuracy"], -metrics["loss"]
        return (metrics["final_answer_accuracy"], *rank) if self.metric == "final_answer_accuracy" else rank

    @property
    def targeted(self):
        return self.target is not None

    def solved(self, metrics):
        return metrics[self.metric] >= self.target
