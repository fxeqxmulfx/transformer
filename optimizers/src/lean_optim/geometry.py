# Copyright (c) 2026 Mikhail Borisov
"""Public matrix optimizers and direction stages, ported from lab at 470f9b6b.

Source equations and deliberate extensions are in LEAN_CORRESPONDENCE.md.
No GPTMini-specific naming, experiment specification or lab module is needed.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from . import fisher, magnitude, matrix, stages
from ._parameters import Parameters, finish, settings
from .config import InverseRoot, NewtonDB
from .direction import DirectionOptimizer

if TYPE_CHECKING:
    from collections.abc import Iterable

    from ._types import Params, Scalar


_DEFAULT_ROOT = NewtonDB()
_MAX_GUARD_SIGMA = 0.5


class Muon(matrix.Muon):
    """Five-step Newton-Schulz Muon; Transformer.Muon.muonStep with wd=0.

    matrix_params selects Muon tensors. Other parameters use bias-corrected
    Adam at auxiliary * lr. A plain iterable defaults to every 2D tensor.
    """

    def __init__(  # noqa: PLR0913 - Preserve the established optimizer/recurrence signature.
        self,
        params: Params,
        lr: float = 1e-3,
        momentum: float = 0.95,
        auxiliary: float = 0.05,
        *,
        matrix_params: Iterable[Tensor] | None = None,
        rate: Scalar | None = None,
    ) -> None:
        """Initialize the parameters, hyperparameters and persistent state."""
        options = settings(lr, momentum=momentum, auxiliary=auxiliary)
        model = Parameters(params, matrix_params)
        super().__init__(options, model, rate)
        finish(self, options, model, rate)


class Dash(matrix.Dash):
    """Blocked Shampoo with raw-Adam norm grafting; Transformer.DASH.graft.

    solver selects the inverse fourth root. Finite iterative solvers give
    approximations; EVD is available for eager execution.
    """

    def __init__(  # noqa: PLR0913, PLR0917 - Preserve the established optimizer/recurrence signature.
        self,
        params: Params,
        lr: float = 1e-3,
        solver: InverseRoot = _DEFAULT_ROOT,
        block: int = 32,
        beta: float = 0.99,
        graft_betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-4,
        *,
        rate: Scalar | None = None,
    ) -> None:
        """Initialize the parameters, hyperparameters and persistent state."""
        if not isinstance(solver, InverseRoot):
            msg = "solver must be a lean_optim inverse-root setting"
            raise TypeError(msg)
        if type(block) is not int or block < 1:
            msg = "block must be a positive integer"
            raise ValueError(msg)
        options = settings(
            lr, solver=solver, block=block, beta=beta, graft_betas=graft_betas, eps=eps
        )
        model = Parameters(params)
        super().__init__(options, model, rate)
        finish(self, options, model, rate)


class AdaFisher(fisher.AdaFisher):
    """Linear-layer Fisher hooks; Transformer.AdaFisher.parameterStep.

    Pass a torch.nn.Module with bias-free Linear layers. Other parameters
    receive identity factors, as in the lab benchmark. Call close() when
    replacing the optimizer to remove its forward hooks.
    """

    def __init__(  # noqa: PLR0913, PLR0917 - Preserve the established optimizer/recurrence signature.
        self,
        model: torch.nn.Module,
        lr: float = 1e-3,
        beta: float = 0.9,
        gamma: float = 0.8,
        damping: float = 1e-3,
        weight_decay: float = 0.0,
        *,
        rate: Scalar | None = None,
    ) -> None:
        """Initialize the parameters, hyperparameters and persistent state."""
        if not isinstance(model, torch.nn.Module):
            msg = "AdaFisher needs a module to collect activation and derivative factors"
            raise TypeError(msg)
        if not 0 < gamma <= 1:
            msg = "gamma must lie in (0, 1]"
            raise ValueError(msg)
        options = settings(lr, beta=beta, gamma=gamma, damping=damping, weight_decay=weight_decay)
        super().__init__(options, model, rate)
        finish(self, options, model, rate)

    def close(self) -> None:
        """Remove the registered Fisher factor hooks."""
        for handle in self.factors.handles:
            handle.remove()
        self.factors.handles.clear()


class AMSGradMD(magnitude.AMSGradMD):
    """Magnitude/direction AMSGrad; Transformer.MagnitudeDirection.amsgradMDProposal.

    Pass matrix_params explicitly to exclude embeddings/readouts. guard_sigma
    enables checkedProposal; only that variant allows direction_rate.
    """

    def __init__(  # noqa: PLR0913, PLR0917 - Preserve the established optimizer/recurrence signature.
        self,
        params: Params,
        lr: float = 1e-3,
        direction_rate: Scalar | None = None,
        gain_rate: float = 1e-3,
        auxiliary_rate: float = 3e-4,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        *,
        matrix_params: Iterable[Tensor] | None = None,
        guard_sigma: float | None = None,
        rate: Scalar | None = None,
    ) -> None:
        """Initialize the parameters, hyperparameters and persistent state."""
        if guard_sigma is not None and not 0 < guard_sigma <= _MAX_GUARD_SIGMA:
            msg = "AMSGradMD guard_sigma must lie in (0, 0.5], as its halved fallback requires"
            raise ValueError(msg)
        if direction_rate is not None and (
            not math.isfinite(direction_rate) or direction_rate <= 0
        ):
            msg = "direction_rate must be finite and positive"
            raise ValueError(msg)
        options = settings(
            lr,
            direction_rate=direction_rate,
            gain_rate=gain_rate,
            auxiliary_rate=auxiliary_rate,
            betas=betas,
            eps=eps,
            _closed_beta2=True,
        )
        model = Parameters(params, matrix_params)
        super().__init__(options, model, rate, guard_sigma)
        finish(self, options, model, rate)


def Guarded(optimizer: DirectionOptimizer, sigma: float = 0.5) -> DirectionOptimizer:  # noqa: N802 - Keep the public compositional constructor name.
    """Apply Transformer.Optimization.descentGuard jointly to all directions."""
    if not isinstance(optimizer, DirectionOptimizer):
        msg = "Guarded needs a direction optimizer; use AdamWDirection for AdamW"
        raise TypeError(msg)
    if not 0 < sigma <= 1 or any(isinstance(stage, stages.Guard) for stage in optimizer.stages):
        msg = "A single guard requires sigma in (0, 1]"
        raise ValueError(msg)
    optimizer.stages.append(stages.Guard(sigma, optimizer))
    return optimizer


def Magma(  # noqa: N802, PLR0913 - Preserve the established optimizer/recurrence signature.
    optimizer: DirectionOptimizer,
    *,
    updates: int,
    seed: int = 0,
    tau: float = 2.0,
    survival: float = 0.5,
    matrix_params: Iterable[Tensor] | None = None,
) -> DirectionOptimizer:
    """Mask/damp matrix directions; Transformer.Magma.algorithmDisplacement.

    The fixed mask plan repeats after updates steps, matching the historical
    graph implementation. No inverse-survival rescaling is performed.
    """
    if not isinstance(optimizer, DirectionOptimizer):
        msg = "Magma needs a direction optimizer"
        raise TypeError(msg)
    if (
        type(updates) is not int
        or updates < 1
        or not math.isfinite(tau)
        or tau <= 0
        or not 0 < survival <= 1
    ):
        msg = "Magma needs positive updates, positive finite tau and survival in (0, 1]"
        raise ValueError(msg)
    if any(isinstance(stage, stages.Magma) for stage in optimizer.stages):
        msg = "Magma may only be applied once"
        raise ValueError(msg)
    original = optimizer.parameter_model
    selected = (
        list(original.hidden_matrices())
        if matrix_params is None and hasattr(original, "hidden_matrices")
        else matrix_params
    )
    model = Parameters(original.parameters(), selected)
    model.named = list(original.named_parameters())
    optimizer.stages.append(
        stages.Magma(SimpleNamespace(tau=tau, survival=survival), optimizer, model, updates, seed)
    )
    return optimizer
