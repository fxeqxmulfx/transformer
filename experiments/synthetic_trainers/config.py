"""Model and training budgets for the reference mini GPT."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ModelSpec:
    width: int = 64
    layers: int = 2
    heads: int = 4
    ff_multiplier: int = 4
    rope_theta: float = 10_000.0

    def __post_init__(self):
        if min(self.width, self.layers, self.heads, self.ff_multiplier) < 1:
            raise ValueError("Model dimensions must be positive")
        if self.width % self.heads or (self.width // self.heads) % 2:
            raise ValueError("RoPE needs an even integer head dimension")
        if not math.isfinite(self.rope_theta) or self.rope_theta <= 0:
            raise ValueError("rope_theta must be finite and positive")

    def reference_config(self, vocab_size, max_length):
        from experiments.gpt_mini import Config

        return Config(vocab_size=vocab_size, n_layers=self.layers, n_heads=self.heads,
                      d_model=self.width, d_ff=self.width * self.ff_multiplier,
                      max_seq_len=max_length, rope_theta=self.rope_theta)


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 200
    batch_size: int = 16
    eval_every: int = 20
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    seed: int = 0
    data_seed: int = 0
    train_examples: int = 512
    validation_examples: int = 128
    test_examples: int = 128
    eval_lengths: tuple[int, ...] = ()
    device: str = "cpu"
    cpu_threads: int = 1
    target: float | None = 0.95
    target_metric: str = "sequence_accuracy"
    stop_at_target: bool = False
    study: str = "standard"
    label_noise: float = 0.0
    noise_seed: int = 0
    split_policy: str = "independent"
    fit_epsilon: float = 0.01
    fit_metric: str = "example_error"
    generalization_patience: int = 2
    curve_tolerance: float = 0.001

    def __post_init__(self):
        if min(self.steps, self.batch_size, self.eval_every, self.train_examples,
               self.validation_examples, self.test_examples, self.cpu_threads, self.generalization_patience) < 1:
            raise ValueError("Budgets and split sizes must be positive")
        if min(self.seed, self.data_seed, self.noise_seed) < 0:
            raise ValueError("Seeds must be nonnegative")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")
        if not math.isfinite(self.grad_clip) or self.grad_clip <= 0:
            raise ValueError("grad_clip must be finite and positive")
        if self.target is not None and (not math.isfinite(self.target) or not 0 <= self.target <= 1):
            raise ValueError("target must be in [0, 1] or None")
        if self.target_metric not in ("token_accuracy", "sequence_accuracy", "balanced_accuracy", "final_answer_accuracy"):
            raise ValueError("Unknown target metric")
        if any(length < 1 for length in self.eval_lengths) or len(set(self.eval_lengths)) != len(self.eval_lengths):
            raise ValueError("Evaluation lengths must be distinct and positive")
        if self.study not in ("standard", "memorization", "double_descent"):
            raise ValueError("Unknown study profile")
        if not math.isfinite(self.label_noise) or not 0 <= self.label_noise <= 1:
            raise ValueError("label_noise must be finite and in [0, 1]")
        if self.study == "standard" and self.label_noise:
            raise ValueError("Label noise requires a memorization or double_descent study")
        if self.split_policy not in ("independent", "disjoint"):
            raise ValueError("Unknown split policy")
        if self.study == "standard" and self.split_policy != "independent":
            raise ValueError("Disjoint pools require a study profile")
        if self.study != "standard" and self.stop_at_target:
            raise ValueError("Study profiles require the full budget to preserve learning curves")
        if not math.isfinite(self.fit_epsilon) or not 0 < self.fit_epsilon < 1:
            raise ValueError("fit_epsilon must be finite and strictly between zero and one")
        if self.fit_metric not in ("example_error", "token_error", "loss"):
            raise ValueError("Unknown interpolation metric")
        if not math.isfinite(self.curve_tolerance) or self.curve_tolerance < 0:
            raise ValueError("curve_tolerance must be finite and nonnegative")
