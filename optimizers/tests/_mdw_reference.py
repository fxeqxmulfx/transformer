"""Allocating references with autograd factor gradients and explicit physical decay.

These calculations do not call optimizer helpers. They implement the
recurrences of Transformer.MDW.Defs and FreeDefs in ordinary tensor arithmetic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Configuration:
    """Variance convention and the paper's effective-rate warmup."""

    adam: bool
    free: bool
    warmup: int = 0
    beta1: float = 0.5
    beta2: float = 0.75
    epsilon: float = 0.125
    decay: float = 0.25


@dataclass
class History:
    """Independent iterates and every mathematical adaptive history."""

    base: torch.Tensor
    evaluation: torch.Tensor
    moment: torch.Tensor
    second: torch.Tensor
    maximum: torch.Tensor
    step: int = 0
    total: float = 0.0

    @classmethod
    def initial(cls, value: torch.Tensor) -> History:
        """Start both iterates at the given point and all histories at zero."""
        return cls(value.clone(), value.clone(), *(torch.zeros_like(value) for _ in range(3)))

    def query(self, configuration: Configuration) -> torch.Tensor:
        """Return the actual latent interpolation used by the oracle."""
        return (1 - configuration.beta1) * self.base + configuration.beta1 * self.evaluation

    def advance(
        self, gradient: torch.Tensor, rate: float, configuration: Configuration
    ) -> tuple[float, float]:
        """Advance genuine moments and the base, without any penalty or projection."""
        self.step += 1
        p = configuration
        self.second = p.beta2 * self.second + (1 - p.beta2) * gradient.square()
        self.maximum = torch.maximum(self.maximum, self.second)
        if p.free:
            warmup = min(1, self.step / p.warmup) if p.warmup else 1
            eta = rate * math.sqrt(1 - p.beta2**self.step) * warmup
            variance = self.second if p.adam else self.maximum
            delta = eta * gradient / (variance.sqrt() + p.epsilon)
            self.total += eta**2
            coefficient = eta**2 / self.total if self.total else 0.0
        else:
            self.moment = p.beta1 * self.moment + (1 - p.beta1) * gradient
            numerator = self.moment / (1 - p.beta1**self.step) if p.adam else self.moment
            variance = self.second / (1 - p.beta2**self.step) if p.adam else self.maximum
            eta, coefficient = rate, 0.0
            delta = eta * numerator / (variance.sqrt() + p.epsilon)
        self.base = self.base - delta
        return eta, coefficient

    def average(self, coefficient: float) -> None:
        """Average only after physical correction and latent storage repair."""
        self.evaluation = (1 - coefficient) * self.evaluation + coefficient * self.base


def gain(value: torch.Tensor, *, smooth: bool) -> torch.Tensor:
    """Use the corresponding implementation's documented softplus convention."""
    return (
        torch.logaddexp(value, torch.zeros_like(value))
        if smooth
        else torch.nn.functional.softplus(value)
    )


def fuse(points: list[torch.Tensor], *, smooth: bool) -> torch.Tensor:
    """Build the actual physical matrix from three independent latent points."""
    direction, row, column = points
    return direction * gain(row, smooth=smooth)[:, None] * gain(column, smooth=smooth)[None, :]


class Reference:
    """A selected matrix and auxiliary vector under independent per-update rates."""

    def __init__(self, values: list[torch.Tensor], configuration: Configuration) -> None:
        """Initialize unit gains, a fixed radius and independent factor histories."""
        self.configuration = configuration
        self.weight = values[0].detach().clone()
        self.radius = self.weight.norm()
        raw = math.log(math.expm1(1))
        self.factors = [
            History.initial(self.weight),
            History.initial(self.weight.new_full((self.weight.size(0),), raw)),
            History.initial(self.weight.new_full((self.weight.size(1),), raw)),
        ]
        self.auxiliary = History.initial(values[1].detach())

    def points(self, *, evaluation: bool = False) -> list[torch.Tensor]:
        """Return real evaluation factors or the actual current oracle query."""
        if evaluation:
            return [factor.evaluation for factor in self.factors]
        if self.configuration.free:
            return [factor.query(self.configuration) for factor in self.factors]
        row, column = self.factors[1:]
        direction = self.weight / (
            gain(row.base, smooth=False)[:, None] * gain(column.base, smooth=False)[None, :]
        )
        return [direction, row.base, column.base]

    def step(self, gradients: list[torch.Tensor], rates: tuple[float, float, float]) -> None:
        """Compute all chain-rule gradients at the old oracle location before any update."""
        p = self.configuration
        points = [point.detach().requires_grad_() for point in self.points()]
        latent_gradients = torch.autograd.grad(
            (fuse(points, smooth=p.free) * gradients[0]).sum(), points
        )
        old = self.weight.clone()
        if not p.free:
            self.factors[0].base = points[0].detach()
        updates = [
            state.advance(gradient, rate, p)
            for state, gradient, rate in zip(
                self.factors, latent_gradients, (rates[0], rates[1], rates[1]), strict=True
            )
        ]
        direction, row, column = self.factors
        eta = updates[0][0]
        if (eta or not p.free) and (rates[0] or rates[1]):
            norm = direction.base.norm()
            direction.base = (
                direction.base * (self.radius / norm) if norm else torch.zeros_like(direction.base)
            )
        candidate = fuse([factor.base for factor in self.factors], smooth=p.free)
        corrected = candidate - eta * p.decay * old
        if eta and p.decay:
            recovered = corrected / (
                gain(row.base, smooth=p.free)[:, None] * gain(column.base, smooth=p.free)[None, :]
            )
            norm = recovered.norm()
            if norm:
                scaled = gain(row.base, smooth=p.free) * norm / self.radius
                row.base = scaled + torch.log(-torch.expm1(-scaled))
                direction.base = recovered * (self.radius / norm)
            else:
                direction.base = torch.zeros_like(recovered)
        if p.free:
            for state, (_, coefficient) in zip(self.factors, updates, strict=True):
                state.average(coefficient)
            self.weight = fuse(self.points(), smooth=True).detach()
        elif rates[0] or rates[1]:
            self.weight = corrected.detach()
        auxiliary_query = self.auxiliary.query(p) if p.free else self.auxiliary.base.clone()
        eta_auxiliary, coefficient = self.auxiliary.advance(gradients[1], rates[2], p)
        self.auxiliary.base = self.auxiliary.base - eta_auxiliary * p.decay * auxiliary_query
        if p.free:
            self.auxiliary.average(coefficient)
