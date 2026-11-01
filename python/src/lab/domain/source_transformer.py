"""The normalized source transformer used by the flat-spectrum formalization."""

from dataclasses import dataclass
import math

from .model import Model
from .spec import require


@dataclass(frozen=True)
class SourceTransformer(Model):
    """diff_gpt GPT at b98c665b, no-cache forward with RoPE in every block.

    Source: review/sequence_model_source.json and
    Transformer.SequenceModel.Attention at 668989c4. Retains
    parameter-free embedding/final and Peri-RMSNorm,
    per-head Q/K RMSNorm, XSA, SwiGLU and tied embedding. Hidden width is
    (8*width+1)//3. Uses the original initialization, float32 RoPE tables,
    dtype RMS epsilon and 1e-12 XSA clipping. Interface deviations: return
    logits alone, omit generation/checkpointing/cache, and allow any prime
    vocabulary (the source's pad_vocab_size_to=1 setting).
    """
    width: int = 128
    depth: int = 1
    heads: int = 4
    context: int = 33
    rope_base: float = 10000.

    def check(self):
        for name in ("width", "depth", "heads", "context"):
            require(type(getattr(self, name)) is int and getattr(self, name) > 0,
                    f"SourceTransformer {name} must be a positive integer")
        require(self.width % (2 * self.heads) == 0,
                "SourceTransformer needs even head dimensions")
        require(math.isfinite(self.rope_base) and self.rope_base > 0,
                "RoPE base must be finite and positive")

    @property
    def hidden(self):
        return (8 * self.width + 1) // 3
