"""Public PyTorch constructors for coordinate rules.

The numerical kernels were moved from lab at commit 470f9b6b. Their source
recurrences and numerical extensions are listed in LEAN_CORRESPONDENCE.md.
Adam uses raw moments, as Transformer.AdamBeyond.adamRule does; AdamW uses
PyTorch's bias-corrected implementation. Joint-direction rules use one group.
"""

import math

import torch

from . import coordinate, direction
from ._parameters import Parameters, finish, settings
from .config import Constant, Decay, Inverse, InverseSqrt


def initialize(optimizer, kernel, params, options, rate=None):
    for key in ("beta1_decay", "lr_decay"):
        if hasattr(options, key) and not isinstance(getattr(options, key), Decay):
            raise TypeError(f"{key} must be a lean_optim decay setting")
    model = Parameters(params)
    kernel.__init__(optimizer, options, model, rate)
    finish(optimizer, options, model, rate)


class SGD(direction.SGD):
    """SGD with decay on the old parameter; Transformer.AdamBeyond.sgdRule."""
    def __init__(self, params, lr=1e-3, weight_decay=0.0, *, rate=None):
        initialize(self, direction.SGD, params, settings(lr, weight_decay=weight_decay), rate)


class AMSGradW(coordinate.AMSGradW):
    """Raw-moment AMSGrad with independent decay; Transformer.AMSGradW.trainingStep."""
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), weight_decay=0.0, eps=1e-8,
                 *, beta1_decay=Constant(), lr_decay=Constant(), rate=None):
        options = settings(lr, betas=betas, weight_decay=weight_decay, eps=eps,
                           beta1_decay=beta1_decay, lr_decay=lr_decay, _closed_beta2=True)
        initialize(self, coordinate.AMSGradW, params, options, rate)


class AMSGrad(AMSGradW):
    """AMSGrad without weight decay; Transformer.AMSGrad.amsgradRule.

    The defaults are constant coefficients. The paper's theorem schedules
    are beta1_decay=Inverse() and lr_decay=InverseSqrt().
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 *, beta1_decay=Constant(), lr_decay=Constant(), rate=None):
        super().__init__(params, lr, betas, 0.0, eps, beta1_decay=beta1_decay, lr_decay=lr_decay, rate=rate)


class Adam(coordinate.Adam):
    """Adam WITHOUT bias correction; Transformer.AdamBeyond.adamRule."""
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 *, beta1_decay=Constant(), lr_decay=Constant(), rate=None):
        initialize(self, coordinate.Adam, params,
                   settings(lr, betas=betas, eps=eps, beta1_decay=beta1_decay, lr_decay=lr_decay), rate)


class AdamX(coordinate.AdamX):
    """Rescaled running maximum; Transformer.AMSGrad.adamXRule (Algorithm 2)."""
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 *, beta1_decay=Inverse(), lr_decay=InverseSqrt(), rate=None):
        initialize(self, coordinate.AdamX, params,
                   settings(lr, betas=betas, eps=eps, beta1_decay=beta1_decay, lr_decay=lr_decay), rate)


class AdaGrad(coordinate.AdaGrad):
    """Mean squared gradients; Transformer.AdamBeyond.adagradRule.

    With InverseSqrt(), eps is added AFTER taking the mean's square root;
    therefore positive eps differs from torch.optim.Adagrad.
    """
    def __init__(self, params, lr=1e-3, eps=1e-8, *, lr_decay=InverseSqrt(), rate=None):
        initialize(self, coordinate.AdaGrad, params, settings(lr, eps=eps, lr_decay=lr_decay), rate)


class AdamNC(coordinate.AdamNC):
    """Momentum over mean squared gradients; Transformer.AdamBeyond.adamNCRule."""
    def __init__(self, params, lr=1e-3, beta1=0.9, eps=1e-8,
                 *, beta1_decay=Inverse(), lr_decay=InverseSqrt(), rate=None):
        initialize(self, coordinate.AdamNC, params,
                   settings(lr, beta1=beta1, eps=eps, beta1_decay=beta1_decay, lr_decay=lr_decay), rate)


class RMSProp(coordinate.RMSProp):
    """Raw second-moment RMSProp without momentum; see the correspondence table."""
    def __init__(self, params, lr=1e-3, beta2=0.999, eps=1e-8, *, rate=None):
        initialize(self, coordinate.RMSProp, params, settings(lr, beta2=beta2, eps=eps), rate)


class AdamW(torch.optim.AdamW):
    """Native PyTorch AdamW; bias-corrected decay model in Landscape.Neural.AdamW.

    All native parameter-group, scheduler and closure behavior is retained.
    """
    def step(self, closure=None):
        if not hasattr(self, "_clip_norm"):
            return super().step(closure)
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        torch.nn.utils.clip_grad_norm_([p for group in self.param_groups for p in group["params"]], self._clip_norm)
        result = super().step()
        return loss if closure is not None else result


class AdamWDirection(coordinate.AdamW):
    """Bias-corrected AdamW expressed as a direction for Guarded/Magma stages.

    Exact-real recurrence matches AdamW; floating-point order differs from
    the native PyTorch kernels. Decay acts on all tensors or only matrices.
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), weight_decay=0.01, eps=1e-8,
                 *, decay="all", rate=None):
        if decay not in ("all", "matrices"):
            raise ValueError("decay must be 'all' or 'matrices'")
        initialize(self, coordinate.AdamW, params,
                   settings(lr, betas=betas, weight_decay=weight_decay, eps=eps, decay=decay), rate)


def Clipped(optimizer, norm=1.0):
    """Clip the joint gradient BEFORE moment updates, with PyTorch's 1e-6 extension.

    Source: Transformer.RecurrentGradients.clip, arXiv:1211.5063 Algorithm 1.
    The real model has no 1e-6. The returned object is the same optimizer.
    """
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError("Clipping norm must be finite and positive")
    if not isinstance(optimizer, (direction.DirectionOptimizer, AdamW)):
        from .geometry import AMSGradMD
        if not isinstance(optimizer, AMSGradMD):
            raise TypeError("Clipped supports direction optimizers, AMSGradMD and native AdamW")
    optimizer._clip_norm = norm
    return optimizer
