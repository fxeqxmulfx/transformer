"""Public matrix optimizers and direction stages, ported from lab at 470f9b6b.

Source equations and deliberate extensions are in LEAN_CORRESPONDENCE.md.
No GPTMini-specific naming, experiment specification or lab module is needed.
"""

from types import SimpleNamespace
import math

import torch

from . import fisher, magnitude, matrix, stages
from ._parameters import Parameters, finish, settings
from .config import InverseRoot, NewtonDB
from .direction import DirectionOptimizer


class Muon(matrix.Muon):
    """Five-step Newton-Schulz Muon; Transformer.Muon.muonStep with wd=0.

    matrix_params selects Muon tensors. Other parameters use bias-corrected
    Adam at auxiliary * lr. A plain iterable defaults to every 2D tensor.
    """
    def __init__(self, params, lr=1e-3, momentum=0.95, auxiliary=0.05, *, matrix_params=None, rate=None):
        options = settings(lr, momentum=momentum, auxiliary=auxiliary)
        model = Parameters(params, matrix_params)
        super().__init__(options, model, rate)
        finish(self, options, model, rate)


class Dash(matrix.Dash):
    """Blocked Shampoo with raw-Adam norm grafting; Transformer.DASH.graft.

    solver selects the inverse fourth root. Finite iterative solvers give
    approximations; EVD is available for eager execution.
    """
    def __init__(self, params, lr=1e-3, solver=NewtonDB(), block=32, beta=0.99,
                 graft_betas=(0.9, 0.999), eps=1e-4, *, rate=None):
        if not isinstance(solver, InverseRoot):
            raise TypeError("solver must be a lean_optim inverse-root setting")
        if type(block) is not int or block < 1:
            raise ValueError("block must be a positive integer")
        options = settings(lr, solver=solver, block=block, beta=beta, graft_betas=graft_betas, eps=eps)
        model = Parameters(params)
        super().__init__(options, model, rate)
        finish(self, options, model, rate)


class AdaFisher(fisher.AdaFisher):
    """Linear-layer Fisher hooks; Transformer.AdaFisher.parameterStep.

    Pass a torch.nn.Module with bias-free Linear layers. Other parameters
    receive identity factors, as in the lab benchmark. Call close() when
    replacing the optimizer to remove its forward hooks.
    """
    def __init__(self, model, lr=1e-3, beta=0.9, gamma=0.8, damping=1e-3, weight_decay=0.0, *, rate=None):
        if not isinstance(model, torch.nn.Module):
            raise TypeError("AdaFisher needs a module to collect activation and derivative factors")
        if not 0 < gamma <= 1:
            raise ValueError("gamma must lie in (0, 1]")
        options = settings(lr, beta=beta, gamma=gamma, damping=damping, weight_decay=weight_decay)
        super().__init__(options, model, rate)
        finish(self, options, model, rate)

    def close(self):
        for handle in self.factors.handles:
            handle.remove()
        self.factors.handles.clear()


class AMSGradMD(magnitude.AMSGradMD):
    """Magnitude/direction AMSGrad; Transformer.MagnitudeDirection.amsgradMDProposal.

    Pass matrix_params explicitly to exclude embeddings/readouts. guard_sigma
    enables checkedProposal; only that variant allows direction_rate.
    """
    def __init__(self, params, lr=1e-3, direction_rate=None, gain_rate=1e-3, auxiliary_rate=3e-4,
                 betas=(0.9, 0.999), eps=1e-8, *, matrix_params=None, guard_sigma=None, rate=None):
        if guard_sigma is not None and not 0 < guard_sigma <= 0.5:
            raise ValueError("AMSGradMD guard_sigma must lie in (0, 0.5], as its halved fallback requires")
        if direction_rate is not None and (not math.isfinite(direction_rate) or direction_rate <= 0):
            raise ValueError("direction_rate must be finite and positive")
        options = settings(lr, direction_rate=direction_rate, gain_rate=gain_rate, auxiliary_rate=auxiliary_rate,
                           betas=betas, eps=eps, _closed_beta2=True)
        model = Parameters(params, matrix_params)
        super().__init__(options, model, rate, guard_sigma)
        finish(self, options, model, rate)


def Guarded(optimizer, sigma=0.5):
    """Apply Transformer.Optimization.descentGuard jointly to all directions."""
    if not isinstance(optimizer, DirectionOptimizer):
        raise TypeError("Guarded needs a direction optimizer; use AdamWDirection for AdamW")
    if not 0 < sigma <= 1 or any(isinstance(stage, stages.Guard) for stage in optimizer.stages):
        raise ValueError("A single guard requires sigma in (0, 1]")
    optimizer.stages.append(stages.Guard(sigma, optimizer))
    return optimizer


def Magma(optimizer, *, updates, seed=0, tau=2.0, survival=0.5, matrix_params=None):
    """Mask/damp matrix directions; Transformer.Magma.algorithmDisplacement.

    The fixed mask plan repeats after updates steps, matching the historical
    graph implementation. No inverse-survival rescaling is performed.
    """
    if not isinstance(optimizer, DirectionOptimizer):
        raise TypeError("Magma needs a direction optimizer")
    if type(updates) is not int or updates < 1 or not math.isfinite(tau) or tau <= 0 or not 0 < survival <= 1:
        raise ValueError("Magma needs positive updates, positive finite tau and survival in (0, 1]")
    if any(isinstance(stage, stages.Magma) for stage in optimizer.stages):
        raise ValueError("Magma may only be applied once")
    original = optimizer._parameter_model
    selected = (list(original.hidden_matrices()) if matrix_params is None and hasattr(original, "hidden_matrices")
                else matrix_params)
    model = Parameters(original.parameters(), selected)
    model.named = list(original.named_parameters())
    optimizer.stages.append(stages.Magma(SimpleNamespace(tau=tau, survival=survival), optimizer, model, updates, seed))
    return optimizer
