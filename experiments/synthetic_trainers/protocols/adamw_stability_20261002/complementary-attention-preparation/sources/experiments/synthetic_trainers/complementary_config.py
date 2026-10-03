"""Explicit complementary-task adaptation of the primary native AdamW recipe.

These controls preserve the optimizer semantics of the modular adaptation of
Convexifying Transformers, Section 4. Task pools, evaluation and generation are
the existing synthetic study protocol, not that manuscript's arithmetic task.
"""

from dataclasses import asdict, dataclass
import math

from .config import TrainConfig
from .scheduled_rates import expected_rate


@dataclass(frozen=True)
class ComplementaryConfig(TrainConfig):
    steps: int = 300000
    eval_every: int = 250
    learning_rate: float = .0003
    weight_decay: float = .1
    grad_clip: float | None = None
    beta2: float = .98
    study: str = "double_descent"
    split_policy: str = "disjoint"
    warmup_steps: int = 10
    learning_rate_schedule: str = "constant"
    anneal_start: int = 150000
    anneal_end: int = 250000
    final_rate_factor: float = .1
    decay_scope: str = "all_trainable_parameters"

    def __post_init__(self):
        super().__post_init__()
        if type(self.steps) is not int or not 0 < self.steps <= 300000:
            raise ValueError("Complementary runs require at most 300000 total updates")
        if (self.optimizer != "adamw" or (self.beta1, self.beta2) != (.9, .98)
                or self.optimizer_epsilon != 1e-8 or self.grad_clip is not None
                or self.decay_scope != "all_trainable_parameters"):
            raise ValueError("Complementary native AdamW must retain the primary optimizer semantics")
        if (self.study != "double_descent" or self.split_policy != "disjoint"
                or self.label_noise != 0 or self.stop_at_target):
            raise ValueError("Complementary studies retain clean disjoint pools and the complete budget")
        if type(self.warmup_steps) is not int or self.warmup_steps < 0:
            raise ValueError("Warmup must be an explicit nonnegative update count")
        if self.learning_rate_schedule not in ("constant", "cosine_tail"):
            raise ValueError("Unknown fixed complementary learning-rate schedule")
        if (not math.isfinite(self.final_rate_factor) or not 0 < self.final_rate_factor < 1
                or type(self.anneal_start) is not int or type(self.anneal_end) is not int
                or not self.warmup_steps <= self.anneal_start < self.anneal_end):
            raise ValueError("Annealing bounds and rate factor must be explicit and valid")
        if self.learning_rate_schedule == "cosine_tail" and self.anneal_end > self.steps:
            raise ValueError("Fixed complementary annealing must fit inside the budget")


def learning_rate(config, completed_steps):
    """Use the same next-update convention as both primary modular trainers."""
    if type(config) is not ComplementaryConfig:
        raise TypeError("Complementary rates require an explicitly tagged config")
    return expected_rate(asdict(config), completed_steps)


def optimizer_description(config):
    return {"name": "adamw", "betas": [config.beta1, config.beta2],
            "epsilon": config.optimizer_epsilon, "bias_correction": True,
            "maximum_second_moment": False, "decoupled_decay": True,
            "decay_scope": config.decay_scope, "gradient_clip": None,
            "warmup_steps": config.warmup_steps,
            "learning_rate_schedule": config.learning_rate_schedule,
            "rate_convention": "next_update_after_completed_steps"}
