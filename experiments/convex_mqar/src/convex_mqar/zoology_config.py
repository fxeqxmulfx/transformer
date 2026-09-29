"""Pinned ICLR 2024 Figure 2 settings, with explicit evaluation changes."""

from dataclasses import dataclass

from .config import Config


SOURCE_COMMIT = "de4e258784224e09909c257ff3ea040f089ed660"
SOURCE_URL = f"https://github.com/HazyResearch/zoology/tree/{SOURCE_COMMIT}"


@dataclass(frozen=True)
class ZoologyConfig(Config):
    seed: int = 123
    data_seed: int = 0
    alpha: float = 0.01
    warmup_fraction: float = 0.
    protocol: str = "zoology-iclr24-figure2"
    positions: str = "rope"
    dropout: float = 0.1
    source_commit: str = SOURCE_COMMIT

    def pair_count(self, length):
        return {64: 4, 128: 8, 256: 16, 512: 64}.get(length, length // 4)

    def batch_size(self, length, width):
        return 64 if length >= 1024 else 128 if length >= 512 else 256 if length >= 256 else 512


def profile(name, positions="rope", device="cuda"):
    common = dict(positions=positions, device=device,
                  precision="bf16" if device == "cuda" else "fp32")
    if name == "full":
        return ZoologyConfig(**common)
    if name == "capacity":
        return ZoologyConfig(widths=(64, 128, 256, 512), **common)
    if name == "smoke":
        return ZoologyConfig(vocab=64, lengths=(16,), widths=(32,),
                             learning_rates=(0.003,), train_examples=256,
                             validation_examples=64, test_examples=64, epochs=3, **common)
    raise ValueError(name)
