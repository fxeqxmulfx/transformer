"""Adapt ordinary PyTorch parameters to the historical numerical kernels.

This module contains no experiment or network architecture dependency.
Matrix membership is explicit; plain iterables default to all 2D tensors.
"""

from types import SimpleNamespace
import math

import torch

from .config import decode, encode


class Parameters:
    def __init__(self, params, matrix_params=None):
        if isinstance(params, torch.nn.Module):
            named = [(name, p) for name, p in params.named_parameters() if p.requires_grad]
        else:
            values = list(params)
            if any(isinstance(p, dict) for p in values):
                raise ValueError("Joint-direction rules take one parameter group; pass a tensor iterable")
            named = [(str(i), p) for i, p in enumerate(values) if p.requires_grad]
        self.named = named
        parameters = [p for _, p in named]
        if not parameters:
            raise ValueError("An optimizer needs trainable parameters")
        if len(set(parameters)) != len(parameters):
            raise ValueError("Duplicate parameters are not supported")
        self.hidden = [p for p in parameters if p.ndim == 2] if matrix_params is None else list(matrix_params)
        if len(set(self.hidden)) != len(self.hidden) or any(p not in set(parameters) or p.ndim != 2 for p in self.hidden):
            raise ValueError("matrix_params must be distinct 2D tensors from params")

    def parameters(self):
        return iter(p for _, p in self.named)

    def named_parameters(self):
        return iter(self.named)

    def hidden_matrices(self):
        return iter(self.hidden)


class Options:
    """Read current hyperparameters, including restored state, from the group."""
    def __init__(self, optimizer):
        self.optimizer = optimizer

    def __getattr__(self, key):
        return decode(self.optimizer.param_groups[0][key])


def finish(optimizer, options, model, rate):
    group = optimizer.param_groups[0]
    group.update({key: encode(value) for key, value in vars(options).items() if key != "lr"})
    group["lr"] = options.lr if rate is None else rate
    optimizer.spec = Options(optimizer)
    optimizer._parameter_model = model


def settings(lr, *, _closed_beta2=False, **kwargs):
    if not math.isfinite(lr) or lr < 0:
        raise ValueError("Learning rate must be finite and nonnegative")
    for key in ("eps", "damping", "gain_rate", "auxiliary_rate", "auxiliary"):
        if key in kwargs and (not math.isfinite(kwargs[key]) or kwargs[key] <= 0):
            raise ValueError(f"{key} must be finite and positive")
    for key in ("weight_decay",):
        if key in kwargs and (not math.isfinite(kwargs[key]) or kwargs[key] < 0):
            raise ValueError(f"{key} must be finite and nonnegative")
    for key in ("beta", "beta1", "beta2", "momentum"):
        if key in kwargs and not 0 <= kwargs[key] < 1:
            raise ValueError(f"{key} must lie in [0, 1)")
    for key in ("betas", "graft_betas"):
        if key in kwargs and (len(kwargs[key]) != 2 or not 0 <= kwargs[key][0] < 1
                              or not (0 <= kwargs[key][1] <= 1 if _closed_beta2 else 0 <= kwargs[key][1] < 1)):
            raise ValueError(f"{key} must contain two numbers in [0, 1)")
    return SimpleNamespace(lr=lr, **kwargs)
