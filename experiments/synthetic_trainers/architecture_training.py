"""Pair the Lean-derived normalizer with an unchanged native AdamW schedule.

The routing projection follows Transformer.GPTMini.Convex.Attention. Modular
division adapts Convexifying Transformers, Section 4. This implementation
does not imply convex joint training, stability, or an architecture effect.
Scientific architecture selection requires the independent benchmark gate.
"""

from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

from .attention_layout import digest
from .paper_reproduction import grokking
from .paper_reproduction.provenance import ROOT
from .scheduled_rates import expected_rate
from .scheduled_training import ScheduledRunConfig, training_sources as scheduled_sources
from .sparsemax_attention import replace_attention


@dataclass(frozen=True)
class ArchitectureRunConfig(ScheduledRunConfig):
    attention_normalization: str = "softmax"

    def __post_init__(self):
        super().__post_init__()
        if type(self.steps) is not int or self.attention_normalization not in ("softmax", "sparsemax"):
            raise ValueError("Architecture pairs require an explicit normalizer and integer update budget")


_original_model = grokking.make_model


def make_model(config, vocab_size):
    if type(config) is not ArchitectureRunConfig:
        raise TypeError("Architecture training requires an explicit ArchitectureRunConfig")
    model = _original_model(config, vocab_size)
    return replace_attention(model) if config.attention_normalization == "sparsemax" else model


def learning_rate(config, completed_steps):
    if type(config) is not ArchitectureRunConfig:
        raise TypeError("Architecture rates require an explicit ArchitectureRunConfig")
    return expected_rate(vars(config), completed_steps)


def training_sources():
    result = scheduled_sources()
    for name in ("architecture_training.py", "sparsemax_attention.py"):
        path = Path(__file__).parent / name
        result[str(path.relative_to(ROOT))] = digest(path)
    return dict(sorted(result.items()))


def train(config, directory, *, resume=False, **kwargs):
    """Preserve optimizer, RNG, batching, diagnostics and native continuation.

    Tags and all nineteen source fingerprints are recorded before checkpoint
    validation/loading. The softmax control retains the exact selected rate
    path; the candidate changes only the causal attention normalization.
    """
    if type(config) is not ArchitectureRunConfig:
        raise TypeError("Architecture training requires an explicit ArchitectureRunConfig")
    with patch.object(grokking, "make_model", make_model), \
            patch.object(grokking, "learning_rate", learning_rate), \
            patch.object(grokking, "source_hashes", training_sources):
        return grokking.train(config, directory, resume=resume, **kwargs)
