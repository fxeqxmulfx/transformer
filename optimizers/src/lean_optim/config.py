"""Pure configuration values for coefficient schedules and inverse roots.

Schedules multiply the coefficient at update t = 1, 2, ... . They mirror
the sequences accepted by Transformer.AMSGrad.Setup. Solver settings mirror
Transformer.DASH's counted inverse-root iterations. See LEAN_CORRESPONDENCE.md.
"""

from __future__ import annotations

from dataclasses import dataclass


class Decay:
    """A coefficient multiplier, with its first value equal to one."""


@dataclass(frozen=True)
class Constant(Decay):
    """c_t = 1."""


@dataclass(frozen=True)
class Inverse(Decay):
    """c_t = 1/t."""


@dataclass(frozen=True)
class InverseSqrt(Decay):
    """c_t = 1/sqrt(t)."""


@dataclass(frozen=True)
class Geometric(Decay):
    """c_t = ratio**(t-1)."""

    ratio: float = 0.99

    def __post_init__(self) -> None:
        """Validate the configuration domain."""
        if not 0 < self.ratio < 1:
            msg = "The geometric ratio must lie in (0, 1)"
            raise ValueError(msg)


class InverseRoot:
    """A batched solver for (A + eps I)**(-1/4)."""


@dataclass(frozen=True)
class EVD(InverseRoot):
    """Eigendecomposition; unsuitable for CUDA graph capture."""


@dataclass(frozen=True)
class NewtonDB(InverseRoot):
    """Two chained Newton-Denman-Beavers square roots."""

    steps: int = 6

    def __post_init__(self) -> None:
        """Validate the configuration domain."""
        if type(self.steps) is not int or self.steps < 1:
            msg = "NewtonDB needs a positive integer iteration count"
            raise ValueError(msg)


@dataclass(frozen=True)
class CoupledNewton(InverseRoot):
    """Coupled Newton iterations for the inverse fourth root."""

    steps: int = 8

    def __post_init__(self) -> None:
        """Validate the configuration domain."""
        if type(self.steps) is not int or self.steps < 1:
            msg = "CoupledNewton needs a positive integer iteration count"
            raise ValueError(msg)


@dataclass(frozen=True)
class Chebyshev(InverseRoot):
    """Cosine fit evaluated by Clenshaw, with at least degree+1 nodes."""

    degree: int = 60
    samples: int = 1000

    def __post_init__(self) -> None:
        """Validate the configuration domain."""
        if (
            type(self.degree) is not int
            or self.degree < 0
            or type(self.samples) is not int
            or self.samples < 1
        ):
            msg = "Chebyshev needs a nonnegative degree and positive sample count"
            raise ValueError(msg)


def encode(value: object) -> object:
    """Keep optimizer state dictionaries limited to ordinary Python values."""
    if isinstance(value, (Decay, InverseRoot)):
        return {"kind": type(value).__name__, **vars(value)}
    return value


def decode(value: object) -> object:
    """Decode a serialized schedule or solver while preserving ordinary values."""
    if isinstance(value, dict) and "kind" in value:
        kinds = (Constant, Inverse, InverseSqrt, Geometric, EVD, NewtonDB, CoupledNewton, Chebyshev)
        cls = next((cls for cls in kinds if cls.__name__ == value["kind"]), None)
        if cls is None:
            msg = f"Unknown optimizer setting {value['kind']!r}"
            raise ValueError(msg)
        return cls(**{key: item for key, item in value.items() if key != "kind"})
    return value
