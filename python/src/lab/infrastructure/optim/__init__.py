"""PyTorch optimizers for optimizer specs.

The native AdamW is constructed as `paper_reproduction.grokking.make_optimizer`
constructed it: every trainable parameter in `parameters()` order, the rate
left for the schedule to set before each update, and PyTorch's default
implementation (foreach on CUDA, a loop over tensors on the CPU). SGD and
AMSGradW are the direction optimizers of `.direction`, with the arithmetic of
the historical `CoordinateOptimizer`.
"""

import torch

from ...domain import optimizers
from . import direction


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


OPTIMIZERS = {optimizers.AdamW: adamw, optimizers.SGD: direction.SGD, optimizers.AMSGradW: direction.AMSGradW}


def build_optimizer(spec, model, rate=None):
    """The optimizer for `spec`; given `rate`, its capturable form.

    `rate` is a one-element tensor on the parameters' device. The capturable
    optimizer keeps all its state there and reads the rate from `rate` when an
    update runs, so a captured update replays with whatever `rate` holds.
    """
    if type(spec) not in OPTIMIZERS:
        raise NotImplementedError(f"No optimizer for {spec!r}")
    return OPTIMIZERS[type(spec)](spec, model, rate)
