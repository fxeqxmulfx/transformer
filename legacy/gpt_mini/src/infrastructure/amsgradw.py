"""AMSGradW with the update formalized in Transformer.AMSGradW.Basic.

Sources: arXiv:1904.03590v4, Algorithm1 and Section6; decoupled decay
from arXiv:2606.25971v2, Sections2/4.1, and the repository's deterministic
AMSGradW extension. This is the raw-moment variant used by the recorded
GPTMini benchmark. It has no bias correction, projection, or step guard.

Only the supplied loss gradient enters the moments. With newly updated
m, v and maximum, the actual parameter step is
    x_next = (1 - lr*weight_decay)*x_old - lr*m/(eps + sqrt(maximum)).

Lean's convergence theorem requires a fixed differentiable objective with
a maximum-norm Lipschitz gradient of constant L, constant positive lr,
eps and decay, lr*decay <=1 and L < decay*eps. The limit is generally a
history-dependent diagonally regularized equilibrium. These objective
hypotheses are not established for stochastic float32 GPTMini training.
"""

from collections.abc import Callable, Iterable
import math
from numbers import Real
from typing import Any

import torch
from torch.optim import Optimizer


@torch.no_grad()
def amsgradw_update_(parameter: torch.Tensor, gradient: torch.Tensor,
                     exp_avg: torch.Tensor, exp_avg_sq: torch.Tensor,
                     max_exp_avg_sq: torch.Tensor, *, lr: float,
                     betas: tuple[float, float], eps: float,
                     weight_decay: float) -> None:
    """Update buffers and old weights in place; AMSGradW.trainingStep.

    Source: arXiv:1904.03590v4, Algorithm1/Section6, decoupled-decay
    extension. The caller supplies distinct buffers matching the parameter.
    All tensor calculations can be included in a full compiler graph.
    """
    beta1, beta2 = betas
    exp_avg.mul_(beta1).add_(gradient, alpha=1 - beta1)
    exp_avg_sq.mul_(beta2).addcmul_(gradient, gradient, value=1 - beta2)
    max_exp_avg_sq.copy_(torch.maximum(max_exp_avg_sq, exp_avg_sq))
    # Read the original parameter for both terms before the single copy.
    parameter.copy_(parameter - lr * exp_avg / (eps + max_exp_avg_sq.sqrt())
                    - (lr * weight_decay) * parameter)


class AMSGradW(Optimizer):
    """Standalone raw-moment AMSGradW for real dense PyTorch parameters.

    Defaults match the winning GPTMini recipe: lr3e-4, betas(.9,.999),
    eps1e-8, decay.01. Parameter groups and optimizer checkpoints use the
    standard PyTorch API. Parameters without a gradient are skipped.

    Example:
        optimizer = AMSGradW(model.parameters())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    """

    def __init__(self, params: Iterable[torch.Tensor] | Iterable[dict[str, Any]],
                 lr: float = 3e-4, betas: tuple[float, float] = (.9, .999),
                 eps: float = 1e-8, weight_decay: float = .01):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self._validate_options(defaults)
        super().__init__(params, defaults)

    @staticmethod
    def _validate_options(options: dict[str, Any]) -> None:
        for name in ("lr", "eps", "weight_decay"):
            value = options[name]
            if not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite real number")
            if value < 0 or name == "eps" and value == 0:
                raise ValueError(f"Invalid {name}: {value}")
        betas = options["betas"]
        if not isinstance(betas, (tuple, list)) or len(betas) != 2:
            raise ValueError("betas must contain two finite coefficients")
        beta1, beta2 = betas
        if (not all(isinstance(b, Real) and math.isfinite(b) for b in betas)
                or not 0 <= beta1 < 1 or not 0 <= beta2 <= 1):
            raise ValueError("Require 0 <= beta1 <1 and 0 <= beta2 <=1")

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        if not isinstance(param_group, dict):
            raise TypeError("Parameter group must be a dict")
        group = dict(param_group)
        raw = group["params"]
        if isinstance(raw, set):
            raise TypeError("Parameters require a deterministic order")
        group["params"] = [raw] if isinstance(raw, torch.Tensor) else list(raw)
        # PyTorch also accepts named parameters. Preserve names for its API.
        tensors = [p[1] if isinstance(p, tuple) else p for p in group["params"]]
        if len({id(p) for p in tensors}) != len(tensors):
            raise ValueError("Duplicate parameters would update tied weights twice")
        for p in tensors:
            if not isinstance(p, torch.Tensor):
                raise TypeError("Optimizer parameters must be tensors")
            if not p.is_floating_point() or p.layout != torch.strided:
                raise ValueError("AMSGradW requires real dense floating-point parameters")
        self._validate_options({name: group.get(name, value) for name, value in self.defaults.items()})
        super().add_param_group(group)
        # Allocate before tracing so the first step need not create state.
        for p in tensors:
            self.state[p] = {name: torch.zeros_like(p, memory_format=torch.preserve_format)
                             for name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq")}

    @torch.no_grad()
    def step(self, closure: Callable[[], torch.Tensor] | None = None) -> torch.Tensor | None:
        """Apply AMSGradW.trainingStep to each unique parameter with a gradient."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        # Reject unsupported gradients before changing any group or history.
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None and p.grad.layout != torch.strided:
                    raise RuntimeError("AMSGradW does not support sparse gradients")
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                amsgradw_update_(p, p.grad, state["exp_avg"], state["exp_avg_sq"],
                    state["max_exp_avg_sq"], lr=group["lr"], betas=group["betas"],
                    eps=group["eps"], weight_decay=group["weight_decay"])
        return loss
