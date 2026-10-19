# Copyright (c) 2026 Mikhail Borisov
"""Types shared by the public constructors and the numerical kernels."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from typing import TYPE_CHECKING, Protocol, TypeAlias

from torch import Tensor, nn

if TYPE_CHECKING:
    from .config import Decay, InverseRoot

Scalar: TypeAlias = float | Tensor
Params: TypeAlias = nn.Module | Iterable[Tensor]
Loss: TypeAlias = Tensor | float | None
Closure: TypeAlias = Callable[[], Loss] | None
TensorState: TypeAlias = dict[str, Tensor]
StateDict: TypeAlias = dict[str, object]
ReportValue: TypeAlias = float | int | bool | str | dict[str, "ReportValue"] | None
Report: TypeAlias = dict[str, ReportValue]


class ParameterModel(Protocol):
    """The parameter interface consumed by the original numerical kernels."""

    def parameters(self) -> Iterator[Tensor]:
        """Iterate over the trainable tensors."""
        ...

    def named_parameters(self) -> Iterator[tuple[str, Tensor]]:
        """Iterate over tensors with stable diagnostic names."""
        ...

    def hidden_matrices(self) -> Iterator[Tensor]:
        """Iterate over the selected matrix parameters."""
        ...


class CoordinateSpec(Protocol):
    """Hyperparameters read by the coordinate kernels of their respective rules."""

    betas: tuple[float, float]
    beta1: float
    beta2: float
    beta1_decay: Decay
    lr_decay: Decay
    eps: float
    weight_decay: float
    decay: str


class MuonSpec(Protocol):
    """Muon momentum and auxiliary Adam rate multiplier."""

    momentum: float
    auxiliary: float


class DashSpec(Protocol):
    """Blocked Shampoo history, graft and inverse-root settings."""

    solver: InverseRoot
    block: int
    beta: float
    graft_betas: tuple[float, float]
    eps: float


class FisherSpec(Protocol):
    """Momentum, Fisher averaging, damping and independent decay."""

    beta: float
    gamma: float
    damping: float
    weight_decay: float


class MagnitudeSpec(Protocol):
    """Independent rates and AMSGrad coefficients of the factorized update."""

    direction_rate: float | None
    gain_rate: float
    auxiliary_rate: float
    betas: tuple[float, float]
    eps: float


class MaskSpec(Protocol):
    """Temperature and survival probability of the fixed MAGMA mask plan."""

    tau: float
    survival: float


class FamilySpec(Protocol):
    """Capacity and original parameter bound of an atomic matching family."""

    cap: float
    heads: int


class MatchingHead(Protocol):
    """Physical Q/K/value tables and the matching map used to interpret them."""

    q: Tensor
    k: Tensor
    values: Tensor
    forward_map: object


class MatchingModel(MatchingHead, Protocol):
    """A simplex mixture supporting column insertion and pruning."""

    spec: FamilySpec
    mass: Tensor
    active: Tensor

    def append(self, head: MatchingHead) -> None:
        """Append a physical matching head within the model's capacity."""
        ...

    def prune(self) -> None:
        """Remove columns of zero simplex mass."""
        ...


class HeadFactory(Protocol):
    """Construct a physical head from original bounded parameter tables."""

    def __call__(
        self, q: Tensor, k: Tensor, values: Tensor, *, forward_map: object = ...
    ) -> MatchingHead:
        """Construct a head interpreted by the supplied matching map."""
        ...


class MatchingTask(Protocol):
    """Observations, supervised forward map and convex output loss."""

    tokens: Tensor

    def forward(
        self, model: MatchingHead, batch: Tensor, *, supervised: bool
    ) -> tuple[Tensor, Tensor]:
        """Return predictions and supervised targets for the selected observations."""
        ...

    def loss(self, output: Tensor, targets: Tensor) -> Tensor:
        """Evaluate the differentiable convex output loss."""
        ...
