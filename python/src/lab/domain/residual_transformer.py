"""Matched residual comparison, separate from the historical legacy-XSA runs."""

from dataclasses import dataclass

from .source_transformer import SourceTransformer
from .modular_composition import ModularPairs
from .spec import require
from .training import MaxAutotune


@dataclass(frozen=True)
class ResidualTransformer(SourceTransformer):
    """Full depth mixing in the normalized GPT host, task145 (2026-11-01).

    Source: Attention Residuals arXiv:2603.15031v1 §3.1 Eqs2--4 and §5;
    diff_gpt ResidualGPT based on 0fab228. Every variant retains centered
    XSA with its clipped remainder and repeated-prefix guard, Q/K RMSNorm,
    Peri-RMSNorm, SwiGLU, exact RoPE and tied readout. 'add' is ordinary
    addition; 'scaled' multiplies each branch by 1/sqrt(2*depth).
    'attnres' uses zero queries and learned key gains as in the paper.
    'qk' instead has unit-RMS queries, parameter-free Q/K normalization and
    a sqrt(width) score divisor; 'conditioned_qk' also changes the complete
    Schedule-Free query coordinates before updates (Lean QueryConditioning).
    Values contain embedding and completed branches once each, including a
    final readout mixture. The singleton first input has no redundant query.
    The host's Peri output normalizations are a disclosed paper deviation.
    Lab interface omits cache/generation/checkpointing and returns logits only.
    """
    mixing: str = "add"

    def check(self):
        super().check()
        require(self.mixing in ("add", "scaled", "attnres", "qk", "conditioned_qk"),
                "Unknown residual mixing")


@dataclass(frozen=True)
class ResidualPairs(ModularPairs):
    """Task145: original training rows, then disjoint validation and final test.

    Split the old held-out rows in half, without resampling. Observe only
    train and validation; the final test is read explicitly after the cohort
    closes. Stop at the current endpoint when train AND validation meet the
    target for consecutive canonical observations after minimum_updates.
    There is no best-checkpoint selection and no final-test-based stopping.
    """
    target_accuracy: float = .99
    stable_observations: int = 3
    minimum_updates: int = 5000

    def check(self):
        super().check()
        size = self.prime * (self.prime - (self.operation == "division"))
        require(size - round(size * self.train_fraction) >= 2,
                "ResidualPairs needs nonempty validation and test splits")
        require(0 < self.target_accuracy <= 1, "Target accuracy must lie in (0,1]")
        require(type(self.stable_observations) is int and self.stable_observations >= 2,
                "Stability needs at least two observations")
        require(type(self.minimum_updates) is int and self.minimum_updates >= 1,
                "Minimum updates must be positive")

    @property
    def observed(self):
        return ("train", "validation")

    def stable(self, history):
        rows = history[-self.stable_observations:]
        return (len(rows) == self.stable_observations and
                rows[0]["step"] >= self.minimum_updates and
                all(row[name]["accuracy"] >= self.target_accuracy
                    for row in rows for name in self.observed))


@dataclass(frozen=True)
class ResidualMaxAutotune(MaxAutotune):
    """Task145 short-context backend selection, still fullgraph max-autotune.

    Source: local Torch2.15 torch.nn.attention.sdpa_kernel/SDPBackend API.
    Math preserves scaled causal softmax attention and ordinary gradients;
    it changes floating-point evaluation and kernel selection. All variants
    in a declared cohort use the same backend; historical runs stay unchanged.
    """
    attention_backend: str = "math"

    def check(self):
        super().check()
        require(self.attention_backend in ("math", "auto"), "Unknown SDPA backend")
