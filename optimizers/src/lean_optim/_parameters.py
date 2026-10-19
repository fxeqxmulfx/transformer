# Copyright (c) 2026 Mikhail Borisov
"""Adapt ordinary PyTorch parameters to the historical numerical kernels.

This module contains no experiment or network architecture dependency.
Matrix membership is explicit; plain iterables default to all 2D tensors.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from .config import decode, encode

MATRIX_NDIM = 2
_BETA_PAIR_LENGTH = 2

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from ._types import ParameterModel, Params, Scalar
    from ._workspace import BufferedOptimizer


class Parameters:
    """Shared update protocol for Parameters."""

    def __init__(self, params: Params, matrix_params: Iterable[Tensor] | None = None) -> None:
        """Initialize the parameters, hyperparameters and persistent state."""
        if isinstance(params, torch.nn.Module):
            named = [(name, p) for name, p in params.named_parameters() if p.requires_grad]
        else:
            values = list(params)
            if any(isinstance(p, dict) for p in values):
                msg = "Joint-direction rules take one parameter group; pass a tensor iterable"
                raise ValueError(msg)
            named = [(str(i), p) for i, p in enumerate(values) if p.requires_grad]
        self.named = named
        parameters = [p for _, p in named]
        if not parameters:
            msg = "An optimizer needs trainable parameters"
            raise ValueError(msg)
        if len(set(parameters)) != len(parameters):
            msg = "Duplicate parameters are not supported"
            raise ValueError(msg)
        self.hidden = (
            [p for p in parameters if p.ndim == MATRIX_NDIM]
            if matrix_params is None
            else list(matrix_params)
        )
        if len(set(self.hidden)) != len(self.hidden) or any(
            p not in set(parameters) or p.ndim != MATRIX_NDIM for p in self.hidden
        ):
            msg = "matrix_params must be distinct 2D tensors from params"
            raise ValueError(msg)

    def parameters(self) -> Iterator[Tensor]:
        """Iterate over trainable tensors in their original order."""
        return iter(p for _, p in self.named)

    def named_parameters(self) -> Iterator[tuple[str, Tensor]]:
        """Iterate over the diagnostic names and trainable tensors."""
        return iter(self.named)

    def hidden_matrices(self) -> Iterator[Tensor]:
        """Iterate over the selected matrix parameters."""
        return iter(self.hidden)


class Options:
    """Read current hyperparameters, including restored state, from the group."""

    def __init__(self, optimizer: BufferedOptimizer) -> None:
        """Initialize the parameters, hyperparameters and persistent state."""
        self.optimizer = optimizer
        self.cache = {}

    def __getattr__(self, key: str) -> object:
        """Read a current hyperparameter, caching decoded configuration values."""
        value = self.optimizer.param_groups[0][key]
        if not isinstance(value, dict):
            return value
        cached = self.cache.get(key)
        if cached is None or cached[0] != value:
            cached = self.cache[key] = (value.copy(), decode(value))
        return cached[1]


def finish(
    optimizer: BufferedOptimizer, options: SimpleNamespace, model: ParameterModel, rate: Scalar
) -> None:
    """Expose current hyperparameters through the serialized parameter group."""
    group = optimizer.param_groups[0]
    group.update({key: encode(value) for key, value in vars(options).items() if key != "lr"})
    group["lr"] = options.lr if rate is None else rate
    optimizer.spec = Options(optimizer)
    optimizer.parameter_model = model


def settings(lr: float, *, _closed_beta2: bool = False, **kwargs: object) -> SimpleNamespace:
    """Validate scalar hyperparameters and collect the kernel settings."""
    if not math.isfinite(lr) or lr < 0:
        msg = "Learning rate must be finite and nonnegative"
        raise ValueError(msg)
    for key in ("eps", "damping", "gain_rate", "auxiliary_rate", "auxiliary"):
        if key in kwargs and (not math.isfinite(kwargs[key]) or kwargs[key] <= 0):
            msg = f"{key} must be finite and positive"
            raise ValueError(msg)
    for key in ("weight_decay",):
        if key in kwargs and (not math.isfinite(kwargs[key]) or kwargs[key] < 0):
            msg = f"{key} must be finite and nonnegative"
            raise ValueError(msg)
    for key in ("beta", "beta1", "beta2", "momentum"):
        if key in kwargs and not 0 <= kwargs[key] < 1:
            msg = f"{key} must lie in [0, 1)"
            raise ValueError(msg)
    for key in ("betas", "graft_betas"):
        if key in kwargs and (
            len(kwargs[key]) != _BETA_PAIR_LENGTH
            or not 0 <= kwargs[key][0] < 1
            or not (0 <= kwargs[key][1] <= 1 if _closed_beta2 else 0 <= kwargs[key][1] < 1)
        ):
            msg = f"{key} must contain two numbers in [0, 1)"
            raise ValueError(msg)
    return SimpleNamespace(lr=lr, **kwargs)
