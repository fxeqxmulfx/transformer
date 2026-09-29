"""Explicit, serializable training budgets for the local MQAR comparison."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Config:
    seed: int = 0
    vocab: int = 8192
    lengths: tuple[int, ...] = (64, 128, 256, 512)
    widths: tuple[int, ...] = (64,)
    learning_rates: tuple[float, ...] = (
        0.0001, 0.00046415888336127773, 0.002154434690031882, 0.01)
    train_examples: int = 100_000
    validation_examples: int = 3000
    test_examples: int = 3000
    epochs: int = 64
    layers: int = 2
    heads: int = 1
    mlp_ratio: int = 4
    weight_decay: float = 0.1
    warmup_fraction: float = 0.1
    alpha: float = 0.1
    rope_base: float = 10_000.0
    device: str = "cuda"
    precision: str = "bf16"

    def batch_size(self, length, width):
        return 8 if max(length, width) >= 512 else 16 if max(length, width) >= 256 else 64

    def to_dict(self):
        return asdict(self)


def profile(name):
    if name == "full":
        return Config()
    if name == "capacity":
        return Config(widths=(64, 128, 256, 512))
    if name == "smoke":
        return Config(vocab=64, lengths=(16,), widths=(32,),
                      learning_rates=(0.003,), train_examples=256,
                      validation_examples=64, test_examples=64, epochs=3)
    if name == "sanity":
        return Config(vocab=16, lengths=(8,), widths=(32,),
                      learning_rates=(0.003,), train_examples=2048,
                      validation_examples=512, test_examples=512, epochs=128)
    raise ValueError(name)
