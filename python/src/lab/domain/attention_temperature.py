"""An isolated learned-temperature intervention on the current normalized GPT."""

from dataclasses import dataclass

from .adaptive_residual import CanonicalResidual


@dataclass(frozen=True)
class TemperatureResidual(CanonicalResidual):
    """Task162: multiply each normalized, rotated query head by exp(log_gain).

    Sources: canonical diff_gpt df1e9f7 and GPTMini QKNorm at bd63e509.
    Deviation from GPTMini: keep current Q/K RMS, residual normalization,
    SwiGLU, initialization and fixed residual scale. Add only a learned
    positive gain per head, initially one, after Q/K normalization and RoPE.
    Zero log-gains consume no randomness and retain the initial function.
    This is an experimental model block, not a production architecture flag.
    """
