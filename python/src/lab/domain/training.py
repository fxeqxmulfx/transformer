"""Training blocks: rate schedule, budget, seeds, cadences, execution."""

from dataclasses import dataclass
import math

from .spec import Spec, require, require_kind


@dataclass(frozen=True)
class Cosine(Spec):
    """Anneal the rate by a half cosine from 1 at `start` to `final` at `end`."""
    start: int
    end: int
    final: float

    def check(self):
        require(0 <= self.start < self.end, "Annealing must end after it starts")
        require(0 < self.final < 1, "The final rate factor lies strictly between zero and one")


@dataclass(frozen=True)
class Schedule(Spec):
    """Linear warmup over `warmup` updates, then constant or annealed.

    An update's warmup factor is min(1, k / warmup), where k counts the
    updates completed before it, so that the first rate is zero; or, with
    `inclusive`, the update itself too, as the convex MQAR trainer counted.
    """
    warmup: int = 0
    anneal: Cosine | None = None
    inclusive: bool = False

    def check(self):
        require(self.warmup >= 0, "Warmup must be nonnegative")
        if self.anneal is not None:
            require_kind(self.anneal, Cosine, "anneal")
            require(self.warmup <= self.anneal.start, "Annealing starts after warmup")


def rate(lr, schedule, completed):
    """The rate of the update that follows `completed` updates.

    The float operations are those of the historical trainers, so the value is
    bit-identical: `paper_reproduction.grokking.learning_rate` without
    annealing, `scheduled_rates.expected_rate` with it, and the convex MQAR
    `train_run` with an inclusive warmup.
    """
    if type(completed) is not int or completed < 0:
        raise TypeError("Completed update count must be a nonnegative integer")
    initial = lr
    if schedule.warmup:
        counted = completed + 1 if schedule.inclusive else completed
        initial *= min(1, counted / max(1, schedule.warmup))
    anneal = schedule.anneal
    if anneal is None or completed <= anneal.start:
        return initial
    fraction = min(1, (completed - anneal.start) / (anneal.end - anneal.start))
    return initial * (anneal.final + (1 - anneal.final) * (1 + math.cos(math.pi * fraction)) / 2)


@dataclass(frozen=True)
class Budget(Spec):
    """`updates` optimizer steps on batches of `batch` training examples, drawn as the benchmark draws them."""
    updates: int
    batch: int

    def check(self):
        require(self.updates >= 1 and self.batch >= 1, "Updates and batch size must be positive")


@dataclass(frozen=True)
class Seeds(Spec):
    """Model initialization, data split, and batch order.

    The batch order seed defaults to model + 10000, the historical convention.
    """
    model: int = 0
    data: int = 0
    batches: int | None = None

    def check(self):
        require(min(self.model, self.data, self.batch_seed) >= 0, "Seeds are nonnegative")

    @property
    def batch_seed(self):
        return self.model + 10_000 if self.batches is None else self.batches


@dataclass(frozen=True)
class Evaluate(Spec):
    """Exhaustive evaluation of every split after each `every` updates."""
    every: int
    batch: int = 1024

    def check(self):
        require(self.every >= 1 and self.batch >= 1, "Evaluation cadence and batch must be positive")


@dataclass(frozen=True)
class Diagnostics(Spec):
    """Extra measurements; none of them changes the trajectory.

    `every` samples per-tensor parameter, gradient, update and moment norms;
    `neighbors` also evaluates and samples the updates one before and one
    after every evaluation; `gradients` records every update's gradient norm.
    """
    every: int = 0
    neighbors: bool = False
    gradients: bool = False

    def check(self):
        require(self.every >= 0, "Diagnostic cadence must be nonnegative")


@dataclass(frozen=True)
class Checkpoint(Spec):
    """Save model, optimizer and sampler state after each `every` updates."""
    every: int = 5000

    def check(self):
        require(self.every >= 1, "Checkpoint cadence must be positive")


@dataclass(frozen=True)
class Execution(Spec, kind=True):
    """How updates are issued to the device."""


@dataclass(frozen=True)
class Eager(Execution):
    """Kernel by kernel with the native optimizer, as the historical trainers ran."""
    device: str = "cuda"
    threads: int = 1

    def check(self):
        require(self.threads >= 1, "Thread count must be positive")


@dataclass(frozen=True)
class CudaGraph(Execution):
    """Each update and each evaluation replays one captured CUDA graph.

    The optimizer runs in its capturable form, whose arithmetic differs from
    the native one in the last bits; a graph run therefore reproduces an eager
    run of the capturable optimizer, not the native trajectory.
    """
    device: str = "cuda"
    threads: int = 1

    def check(self):
        require(self.device.startswith("cuda"), "CUDA graphs need a CUDA device")
        require(self.threads >= 1, "Thread count must be positive")


@dataclass(frozen=True)
class Compiled(Execution):
    """Each update's forward and loss, and each evaluation's forward, compiled by TorchInductor.

    An update reads its batch at the full width of the training split, so
    each batch size compiles once, and the native AdamW runs its fused
    kernel. Compiled kernels and the fused optimizer round differently from
    the eager ones in the last bits: a compiled run reproduces itself at its
    thread count, not an Eager run.
    """
    device: str = "cpu"
    threads: int = 1

    def check(self):
        require(self.threads >= 1, "Thread count must be positive")
