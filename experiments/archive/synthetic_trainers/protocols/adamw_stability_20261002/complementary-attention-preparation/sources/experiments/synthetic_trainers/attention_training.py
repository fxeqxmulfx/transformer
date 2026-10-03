"""Use the unchanged modular AdamW trainer with an explicit normalizer tag.

The sparsemax model follows Transformer.GPTMini.Convex.Attention; the modular
task adapts Convexifying Transformers, Section 4. These are implementation
choices, not a claim of convex joint training or stable generalization.
"""

from dataclasses import dataclass
import hashlib
from pathlib import Path
from unittest.mock import patch

from .paper_reproduction import grokking
from .paper_reproduction.provenance import ROOT, source_hashes
from .sparsemax_attention import replace_attention


@dataclass(frozen=True)
class AttentionRunConfig(grokking.RunConfig):
    attention_normalization: str = "softmax"

    def __post_init__(self):
        super().__post_init__()
        if self.model != "gptmini" or self.optimizer != "adamw":
            raise ValueError("Normalizer controls require GPTMini and native AdamW")
        if self.attention_normalization not in ("softmax", "sparsemax"):
            raise ValueError("Unknown attention normalization")
        if self.steps > 300000:
            raise ValueError("The user cap is 300,000 total updates per run")


_original_make_model = grokking.make_model


def make_model(config, vocab_size):
    if type(config) is not AttentionRunConfig:
        raise TypeError("Normalizer training requires an explicit AttentionRunConfig")
    model = _original_make_model(config, vocab_size)
    return replace_attention(model) if config.attention_normalization == "sparsemax" else model


def training_sources():
    """Pin both control paths, including this factory and support derivative."""
    hashes = source_hashes()
    for name in ("attention_training.py", "sparsemax_attention.py"):
        path = Path(__file__).parent / name
        hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return dict(sorted(hashes.items()))


def train(config, directory, *, resume=False, **kwargs):
    """Tag configs and source hashes before the core validates/loads a resume.

    All original optimization, sampling, timing, diagnostics and checkpoint
    behavior is retained. Serial callers scope the factory change to this call;
    even an exception restores the original module functions.
    """
    if type(config) is not AttentionRunConfig:
        raise TypeError("Normalizer training requires an explicit AttentionRunConfig")
    with patch.object(grokking, "make_model", make_model), patch.object(grokking, "source_hashes", training_sources):
        return grokking.train(config, directory, resume=resume, **kwargs)
