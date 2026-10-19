# Copyright (c) 2026 Mikhail Borisov
"""Inverse fourth roots of regularized Shampoo histories, batched over blocks.

Ports of `optimizer_benchmark.roots`, with its arithmetic. Each solver takes
a stack of histories A and the regularization eps and returns
(A + eps I)^(-1/4) for every block. The Newton solvers and the Chebyshev fit
work on A scaled into the unit interval by `guarded_scale`, after
`Transformer.DASH.countedBatchPiInverseFourth`, `countedBatchPiCnFour` and
`countedBatchPiPower`, and restore the scale on the way out.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from . import config as optimizers
from ._root_workspace import InverseFourth

if TYPE_CHECKING:
    from .config import InverseRoot


def rayleigh_estimate(matrix: Tensor, steps: int = 3) -> Tensor:
    """Compute the Rayleigh quotient after power iterations from the ones vector."""
    vector = torch.ones((*matrix.shape[:-1], 1), dtype=matrix.dtype, device=matrix.device)
    for _ in range(steps):
        vector = matrix @ vector
        norm = vector.square().sum(dim=-2, keepdim=True).sqrt()
        vector = vector / torch.where(norm > 0, norm, torch.ones_like(norm))
    numerator = (vector * (matrix @ vector)).sum(dim=(-2, -1))
    denominator = vector.square().sum(dim=(-2, -1))
    return numerator / torch.where(denominator > 0, denominator, torch.ones_like(denominator))


def guarded_scale(matrix: Tensor) -> Tensor:
    """Use twice a credible Rayleigh estimate or norm bound, and one at zero."""
    frobenius = matrix.square().sum(dim=(-2, -1)).sqrt()
    row_bound = matrix.abs().sum(dim=-1).amax(dim=-1)
    upper = torch.minimum(frobenius, row_bound)
    estimate = rayleigh_estimate(matrix)
    return torch.where(
        (estimate > 0) & (upper < 2 * estimate),
        2 * estimate,
        torch.where(upper > 0, 2 * upper, torch.ones_like(upper)),
    )


def ndb_iterate(matrix: Tensor, steps: int) -> tuple[Tensor, Tensor]:
    """Newton-Denman-Beavers: the square root and its inverse after `steps` iterations."""
    identity = torch.eye(matrix.size(-1), dtype=matrix.dtype, device=matrix.device)
    root, inverse = matrix, identity.expand_as(matrix)
    if steps:
        correction = 1.5 * identity - 0.5 * matrix
        root, inverse = matrix @ correction, correction
    for _ in range(1, steps):
        correction = 1.5 * identity - 0.5 * (inverse @ root)
        root, inverse = root @ correction, correction @ inverse
    return root, inverse


def regularized(history: Tensor, eps: float) -> Tensor:
    """Add isotropic damping to each history matrix."""
    return history + eps * torch.eye(history.size(-1), device=history.device, dtype=history.dtype)


def newton_db(history: Tensor, eps: float, steps: int) -> Tensor:
    """Compute the inverse square root of the square root by two chained NDB runs."""
    matrix = regularized(history, eps)
    scale = guarded_scale(matrix)
    first_root, _ = ndb_iterate(matrix / scale[..., None, None], steps)
    _, inverse = ndb_iterate(first_root, steps)
    return inverse * scale.pow(-0.25)[..., None, None]


def coupled_newton(history: Tensor, eps: float, steps: int) -> Tensor:
    """Coupled Newton for the inverse fourth root, from the identity at the unit scale 5^(-1/4)."""
    matrix = regularized(history, eps)
    identity = torch.eye(matrix.size(-1), dtype=matrix.dtype, device=matrix.device)
    scale = guarded_scale(matrix)
    normalized = matrix / scale[..., None, None]
    unit_scale = 5 ** (-0.25)
    x = identity.expand_as(matrix) / unit_scale
    residual = normalized / unit_scale**4
    for _ in range(steps):
        correction = 1.25 * identity - 0.25 * residual
        x = x @ correction
        squared = correction @ correction
        residual = (squared @ squared) @ residual
    return x * scale.pow(-0.25)[..., None, None]


def eigendecomposition(history: Tensor, eps: float) -> Tensor:
    """Evaluate the regularized inverse fourth root by symmetric eigendecomposition."""
    matrix = regularized(history, eps)
    values, vectors = torch.linalg.eigh(matrix)
    # Histories are regularized and positive definite; the clamp only guards roundoff.
    values = values.clamp_min(torch.finfo(matrix.dtype).tiny)
    return (vectors * values.pow(-0.25).unsqueeze(-2)) @ vectors.transpose(-2, -1)


class Chebyshev:
    """Fit x -> (eps / scale + x)^(-1/4) and evaluate it by Clenshaw.

    The nodes and the cosine basis are computed once per device and dtype, at
    the first call; a graph warm-up makes that call before any capture.
    """

    def __init__(self, degree: int, samples: int, exponent: float = -0.25) -> None:
        """Initialize the parameters, hyperparameters and persistent state."""
        self.degree, self.samples, self.exponent = degree, max(samples, degree + 1), exponent
        self.bases = {}

    def basis(self, matrix: Tensor) -> tuple[Tensor, Tensor]:
        """Return cached cosine nodes and their mode basis."""
        key = (str(matrix.device), matrix.dtype)
        if key not in self.bases:
            angles = 2 * torch.arange(self.samples, dtype=matrix.dtype, device=matrix.device) + 1
            angles = angles * (math.pi / (2 * self.samples))
            modes = torch.arange(self.degree + 1, dtype=matrix.dtype, device=matrix.device)
            self.bases[key] = angles.cos(), (angles[:, None] * modes).cos()
        return self.bases[key]

    def __call__(self, matrix: Tensor, eps: float) -> Tensor:
        """Evaluate the Chebyshev approximation on a batch of histories."""
        degree, exponent = self.degree, self.exponent
        scale = guarded_scale(matrix)
        identity = torch.eye(matrix.size(-1), dtype=matrix.dtype, device=matrix.device)
        argument = 2 * matrix / scale[..., None, None] - identity
        nodes, basis = self.basis(matrix)
        values = (eps / scale[..., None] + (nodes + 1) / 2).pow(exponent)
        coefficients = (2 / self.samples) * (values @ basis)
        coefficients[..., 0] *= 0.5
        if degree == 0:
            output = coefficients[..., 0, None, None] * identity
        elif degree == 1:
            output = coefficients[..., 1, None, None] * argument
            output = output + coefficients[..., 0, None, None] * identity
        else:
            second = coefficients[..., degree, None, None] * identity
            first = 2 * coefficients[..., degree, None, None] * argument
            first = first + coefficients[..., degree - 1, None, None] * identity
            for k in range(degree - 2, 0, -1):
                first, second = (
                    (2 * (argument @ first) - second + coefficients[..., k, None, None] * identity),
                    first,
                )
            output = argument @ first - second + coefficients[..., 0, None, None] * identity
        return output * scale.pow(exponent)[..., None, None]


def build_root(spec: InverseRoot) -> InverseFourth:
    """(history, eps) -> (history + eps I)^(-1/4), batched, for an `InverseRoot` block."""
    if isinstance(
        spec, (optimizers.NewtonDB, optimizers.CoupledNewton, optimizers.EVD, optimizers.Chebyshev)
    ):
        return InverseFourth(spec)
    msg = f"No builder for {spec!r}"
    raise NotImplementedError(msg)
