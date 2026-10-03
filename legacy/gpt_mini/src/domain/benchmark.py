"""Plan paired optimizer comparisons using frozen validation-selected rates."""

from dataclasses import dataclass
import math
import random

from .stopping import StopConfig


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 65
    n_layers: int = 2
    n_heads: int = 4
    d_model: int = 128
    d_ff: int = 512
    max_seq_len: int = 64
    rope_theta: float = 10000.

    def __post_init__(self):
        counts = (self.vocab_size, self.n_layers, self.n_heads, self.d_model, self.d_ff, self.max_seq_len)
        if any(type(n) is not int or n <= 0 for n in counts):
            raise ValueError("Model dimensions must be positive integers")
        if self.d_model % self.n_heads or self.d_model // self.n_heads % 2:
            raise ValueError("RoPE requires an even head width dividing the model width")
        if not math.isfinite(self.rope_theta) or self.rope_theta <= 0:
            raise ValueError("RoPE base must be finite and positive")


@dataclass(frozen=True)
class Request:
    methods: tuple[str, ...] = ("amsgradw",)
    attentions: tuple[str, ...] = ("softmax", "sparsemax")
    seeds: tuple[int, ...] = (0, 1, 2)
    batch: int = 32
    model: ModelConfig = ModelConfig()
    stopping: StopConfig = StopConfig()
    tests_only: bool = False
    validate_only: bool = False
    allow_partial: bool = False

    def __post_init__(self):
        if not self.attentions or len(set(self.attentions)) != len(self.attentions) or not set(self.attentions) <= {"softmax", "sparsemax"}:
            raise ValueError("Select distinct Softmax/Sparsemax attention modes")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds) or any(type(n) is not int or n < 0 for n in self.seeds):
            raise ValueError("Select distinct nonnegative seeds")
        if type(self.batch) is not int or self.batch <= 0:
            raise ValueError("Batch size must be a positive integer")
        if len(set(self.methods)) != len(self.methods):
            raise ValueError("Optimizer names must be distinct")
        if self.tests_only and self.validate_only:
            raise ValueError("Choose tests or checkpoint validation")


@dataclass(frozen=True)
class Job:
    attention: str
    method: str
    rate: float
    seed: int

    @property
    def identifier(self):
        return f"{self.attention}:{self.method}:{self.seed}"


def plan_jobs(request, rates):
    methods = request.methods or tuple(sorted(set.intersection(*(set(rates[a]) for a in request.attentions))))
    for attention in request.attentions:
        for method in methods:
            if method not in rates[attention] or not math.isfinite(rates[attention][method]) or rates[attention][method] <= 0:
                raise ValueError(f"No valid selected rate for {attention}/{method}")
    groups = [(attention, method) for attention in request.attentions for method in methods]
    random.Random(1729).shuffle(groups)
    return tuple(Job(a, m, rates[a][m], seed) for a, m in groups for seed in request.seeds)
