"""The current GPT and a single conditional-amplitude research candidate."""

from dataclasses import dataclass
from typing import ClassVar

from .residual_transformer import ResidualTransformer


@dataclass(frozen=True)
class CanonicalResidual(ResidualTransformer):
    """diff_gpt GPT at df1e9f7, including the task155 accurate RMS backward.

    Fixed post-RMS coefficient 1/sqrt(2*depth), QK RMS, centered XSA,
    all-layer RoPE and tied readout. Interface deviations are the historical
    lab port's logits-only return and omission of cache and checkpointing.
    The inherited historical mixing option is not a constructor field.
    """
    mixing: ClassVar[str] = "scaled"


@dataclass(frozen=True)
class AdaptiveResidual(CanonicalResidual):
    """Task158 candidate: input-dependent gains AFTER each branch output RMS.

    Source: experiments/context_gains/README.md Eqs. (1)-(3), extending the
    Lean Transformer.Residual.gain_derivative_ratio witness to a conditional
    gate. Deviation: the full transformer supplies its normalized branch
    input, rather than the synthetic two-feature vector. Add a bias-free
    linear row per branch and multiply by 2*sigmoid(row(input)). Zero rows
    preserve the initial forward and every common parameter gradient.
    This is an experimental hypothesis, not a faster-grokking theorem.
    """
