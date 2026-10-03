"""Training blocks: optimizer, rate schedule, budget, seeds, cadences, execution."""

from dataclasses import dataclass
import math

from .spec import Spec, require, require_kind


@dataclass(frozen=True)
class Optimizer(Spec, kind=True):
    """The update rule."""


@dataclass(frozen=True)
class AdamW(Optimizer):
    """torch.optim.AdamW: bias-corrected moments and decoupled decay.

    `decay` selects the decayed tensors: "all" trainable parameters, or only
    the "matrices" (ndim >= 2) with vectors in an undecayed group.
    """
    lr: float
    betas: tuple[float, float]
    weight_decay: float
    eps: float = 1e-8
    decay: str = "all"

    def check(self):
        require(math.isfinite(self.lr) and self.lr > 0, "Learning rate must be positive")
        require(len(self.betas) == 2 and all(0 <= beta < 1 for beta in self.betas),
                "AdamW needs two betas in [0, 1)")
        require(self.weight_decay >= 0 and self.eps > 0, "Decay must be nonnegative, epsilon positive")
        require(self.decay in ("all", "matrices"), "AdamW decay scope is 'all' or 'matrices'")


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
    """Linear warmup over `warmup` updates, then constant or annealed."""
    warmup: int = 0
    anneal: Cosine | None = None

    def check(self):
        require(self.warmup >= 0, "Warmup must be nonnegative")
        if self.anneal is not None:
            require_kind(self.anneal, Cosine, "anneal")
            require(self.warmup <= self.anneal.start, "Annealing starts after warmup")


def rate(lr, schedule, completed):
    """The rate of the update that follows `completed` updates.

    The float operations are those of the historical trainers, so the value is
    bit-identical: `paper_reproduction.grokking.learning_rate` without
    annealing and `scheduled_rates.expected_rate` with it.
    """
    if type(completed) is not int or completed < 0:
        raise TypeError("Completed update count must be a nonnegative integer")
    initial = lr
    if schedule.warmup:
        initial *= min(1, completed / max(1, schedule.warmup))
    anneal = schedule.anneal
    if anneal is None or completed <= anneal.start:
        return initial
    fraction = min(1, (completed - anneal.start) / (anneal.end - anneal.start))
    return initial * (anneal.final + (1 - anneal.final) * (1 + math.cos(math.pi * fraction)) / 2)


@dataclass(frozen=True)
class Budget(Spec):
    """`updates` optimizer steps on batches of `batch` training examples.

    Batches walk shuffled epochs. With tail "short" an epoch ends with its
    remainder as a smaller batch; with "wrap" a batch fills across epochs.
    """
    updates: int
    batch: int
    tail: str = "short"

    def check(self):
        require(self.updates >= 1 and self.batch >= 1, "Updates and batch size must be positive")
        require(self.tail in ("short", "wrap"), "Batch tail policy is 'short' or 'wrap'")


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
