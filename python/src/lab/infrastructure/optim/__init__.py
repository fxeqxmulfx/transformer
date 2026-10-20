"""Translate lab optimizer specifications into the standalone lean_optim API.

Numerical implementations live in optimizers/, installable on its own. This
adapter preserves native AdamW selection, the initial rate of zero, hidden
matrix selection, stage ordering and seeds. Steppers clip gradients once.
"""

from dataclasses import fields
from contextlib import contextmanager

import lean_optim
from lean_optim import coordinate

from ...domain import optimizers
from ...domain.mdw import AMSGradMDW
from ...domain.schedule_free import ScheduleFreeAdamW, ScheduleFreeMuon
from .ansr import ANSR


def setting(value):
    if isinstance(value, (optimizers.Decay, optimizers.InverseRoot)):
        return getattr(lean_optim, type(value).__name__)(**vars(value))
    return value


def parameter_groups(spec, model):
    parameters = [p for p in model.parameters() if p.requires_grad]
    if spec.decay == "all":
        return parameters
    return [{"params": [p for p in parameters if p.ndim >= 2]},
            {"params": [p for p in parameters if p.ndim < 2], "weight_decay": 0.0}]


def rule(spec, model, rate, updates, seed):
    if isinstance(spec, optimizers.Guarded):
        if isinstance(spec.base, optimizers.AMSGradMD):
            return construct(spec.base, model, rate, guard_sigma=spec.sigma)
        return lean_optim.Guarded(rule(spec.base, model, rate, updates, seed), spec.sigma)
    if isinstance(spec, optimizers.Magma):
        if updates is None or (spec.seed is None and seed is None):
            raise ValueError("MAGMA draws the masks of the run's updates from its seed or the model's")
        optimizer = rule(spec.base, model, rate, updates, seed)
        return lean_optim.Magma(optimizer, updates=updates, seed=seed + 20_000 if spec.seed is None else spec.seed,
                               tau=spec.tau, survival=spec.survival, matrix_params=list(model.hidden_matrices()))
    return construct(spec, model, rate)


def construct(spec, model, rate, **extra):
    options = {field.name: setting(getattr(spec, field.name)) for field in fields(spec)
               if field.name != "schedule_components"}
    name = "AdamWDirection" if isinstance(spec, optimizers.AdamW) else type(spec).__name__
    cls = getattr(lean_optim, name)
    if isinstance(spec, ScheduleFreeMuon):
        matrices = getattr(model, "hidden_matrices", None)
        options["matrix_params"] = list(matrices()) if matrices is not None else None
    elif isinstance(spec, (optimizers.Muon, optimizers.AMSGradMD, AMSGradMDW)):
        options["matrix_params"] = list(model.hidden_matrices())
    if isinstance(spec, (ScheduleFreeAdamW, ScheduleFreeMuon)):
        if rate is not None:
            raise ValueError("Schedule-Free uses eager optimizer steps without tensor rates")
        optimizer = cls(model, **options, **extra)
    else:
        optimizer = cls(model, **options, rate=rate, **extra)
    if isinstance(spec, (optimizers.AMSGradMD, AMSGradMDW)) and spec.schedule_components:
        optimizer.param_groups[0]["initial_lr"] = spec.lr
    if rate is None:
        optimizer.param_groups[0]["lr"] = 0.0
    return optimizer


def report(optimizer):
    return optimizer.report() if hasattr(optimizer, "report") else {}


def training_weights(optimizer):
    """Restore a Schedule-Free query before a training forward when needed."""
    if isinstance(optimizer, lean_optim.ScheduleFreeSGD) and not optimizer.param_groups[0]["training"]:
        optimizer.train()


@contextmanager
def evaluation_weights(optimizer):
    """Install Schedule-Free averages for both training and held-out evaluation."""
    free = isinstance(optimizer, lean_optim.ScheduleFreeSGD)
    training = free and optimizer.param_groups[0]["training"]
    if free:
        optimizer.eval()
    try:
        yield
    finally:
        if training:
            optimizer.train()


def build_optimizer(spec, model, rate=None, updates=None, seed=None, fused=False):
    if isinstance(spec, optimizers.Clipped):
        spec = spec.base
    if isinstance(spec, optimizers.ANSR):
        if rate is not None or fused:
            raise ValueError("ANSR requires eager population evaluation without a learning rate")
        return ANSR(spec, model, seed)
    if isinstance(spec, optimizers.AdamW):
        options = dict(betas=spec.betas, eps=spec.eps, weight_decay=spec.weight_decay)
        if rate is not None:
            options.update(foreach=True, capturable=True)
        elif fused:
            options["fused"] = True
        return lean_optim.AdamW(parameter_groups(spec, model), lr=0.0 if rate is None else rate, **options)
    return rule(spec, model, rate, updates, seed)
