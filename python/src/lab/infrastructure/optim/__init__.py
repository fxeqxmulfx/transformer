"""PyTorch optimizers for optimizer specs.

The native AdamW is constructed as `paper_reproduction.grokking.make_optimizer`
constructed it: every trainable parameter in `parameters()` order, the rate
left for the schedule to set before each update, and PyTorch's default
implementation (foreach on CUDA, a loop over tensors on the CPU). Every other
rule is a direction optimizer (`.direction`), with the arithmetic of the
historical optimizer zoo, and so is AdamW under a stage (`.stages`), which
transforms the directions of a rule. AMSGradMD (`.magnitude`) writes new
values instead, and carries its own guard.
"""

import torch

from ...domain import optimizers
from . import coordinate, direction, fisher, magnitude, matrix, stages


def parameter_groups(spec, model):
    """One group, or decayed matrices apart from undecayed vectors."""
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if spec.decay == "all":
        return parameters
    return [{"params": [parameter for parameter in parameters if parameter.ndim >= 2]},
            {"params": [parameter for parameter in parameters if parameter.ndim < 2], "weight_decay": 0.0}]


def adamw(spec, model, rate=None):
    groups = parameter_groups(spec, model)
    if rate is None:
        return torch.optim.AdamW(groups, lr=0.0, betas=spec.betas, eps=spec.eps, weight_decay=spec.weight_decay)
    return torch.optim.AdamW(groups, lr=rate, betas=spec.betas, eps=spec.eps, weight_decay=spec.weight_decay,
                             foreach=True, capturable=True)


RULES = {optimizers.SGD: direction.SGD, optimizers.AdamW: coordinate.AdamW, optimizers.AMSGradW: coordinate.AMSGradW,
         optimizers.Adam: coordinate.Adam, optimizers.AdamX: coordinate.AdamX, optimizers.AdaGrad: coordinate.AdaGrad,
         optimizers.AdamNC: coordinate.AdamNC, optimizers.RMSProp: coordinate.RMSProp, optimizers.Muon: matrix.Muon,
         optimizers.Dash: matrix.Dash, optimizers.AdaFisher: fisher.AdaFisher,
         optimizers.AMSGradMD: magnitude.AMSGradMD}


def rule(spec, model, rate, updates, seed):
    """The optimizer of `spec`, with the stages it names applied innermost first."""
    if isinstance(spec, optimizers.Guarded) and isinstance(spec.base, optimizers.AMSGradMD):
        return magnitude.AMSGradMD(spec.base, model, rate, spec.sigma)
    if isinstance(spec, optimizers.Guarded):
        optimizer = rule(spec.base, model, rate, updates, seed)
        optimizer.stages.append(stages.Guard(spec.sigma, optimizer))
        return optimizer
    if isinstance(spec, optimizers.Magma):
        if updates is None or (spec.seed is None and seed is None):
            raise ValueError("MAGMA draws the masks of the run's updates from its seed or the model's")
        optimizer = rule(spec.base, model, rate, updates, seed)
        mask_seed = seed + 20_000 if spec.seed is None else spec.seed
        optimizer.stages.append(stages.Magma(spec, optimizer, model, updates, mask_seed))
        return optimizer
    if type(spec) not in RULES:
        raise NotImplementedError(f"No optimizer for {spec!r}")
    return RULES[type(spec)](spec, model, rate)


def report(optimizer):
    """What the optimizer and its stages counted; nothing for a native optimizer."""
    return optimizer.report() if hasattr(optimizer, "report") else {}


def build_optimizer(spec, model, rate=None, updates=None, seed=None):
    """The optimizer for `spec`; given `rate`, its capturable form.

    `rate` is a one-element tensor on the parameters' device. The capturable
    optimizer keeps all its state there and reads the rate from `rate` when an
    update runs, so a captured update replays with whatever `rate` holds.
    `updates` and `seed`, the run's budget and model seed, are what MAGMA
    draws its masks for and from. A stepper clips the gradient
    (`optimizers.clipping`) before the optimizer steps.
    """
    if isinstance(spec, optimizers.Clipped):
        spec = spec.base
    if isinstance(spec, optimizers.AdamW):
        return adamw(spec, model, rate)
    return rule(spec, model, rate, updates, seed)
