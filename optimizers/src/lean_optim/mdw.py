"""Adam/AMSGrad MD with decoupled physical-weight decay.

Source: local AdamW extension of arXiv:2606.25971v2, Appendix A, Algorithm 2.
Lean: Transformer.MDW.stepWith, adamMDWProposal and amsgradMDWProposal.
The fused proposal loses rate * weight_decay * old_weight; repairing its
row gains preserves that physical weight. Auxiliary coordinates follow
the same adaptive rule and decay at their own effective learning rate.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ._md_adam import adam
from ._mdw_decay import PhysicalDecay
from ._parameters import settings
from ._rates import active_rates, component_rates, copy_active
from .geometry import AMSGradMD

if TYPE_CHECKING:
    from collections.abc import Iterable

    from torch import Tensor

    from ._types import Closure, Loss, Params, Scalar, StateDict


class AMSGradMDW(AMSGradMD):
    """Raw-moment AMSGradMD followed by independent physical-weight decay.

    Moments use loss gradients only. The direction radius survives storage
    repair for nonzero results. Zero corrected matrices keep finite gains.
    A scheduler scales all component rates; decay follows the direction or
    auxiliary rate of its physical tensor. This local composition has no
    general convergence guarantee. Gradient clipping remains supported.
    """

    _bias_corrected = False

    def __init__(  # noqa: PLR0913 - Match the MD family's established hyperparameter interface.
        self,
        params: Params,
        lr: float = 1e-3,
        *,
        gain_rate: float = 1e-3,
        auxiliary_rate: float = 3e-4,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        matrix_params: Iterable[Tensor] | None = None,
        rate: Scalar | None = None,
    ) -> None:
        """Initialize factor memories, nominal rates and physical decay buffers."""
        settings(lr, betas=betas, weight_decay=weight_decay, _closed_beta2=not self._bias_corrected)
        super().__init__(
            params,
            lr=lr,
            gain_rate=gain_rate,
            auxiliary_rate=auxiliary_rate,
            betas=betas,
            eps=eps,
            matrix_params=matrix_params,
            rate=rate,
        )
        self.param_groups[0].update(
            weight_decay=weight_decay,
            mdw_rule="adam" if self._bias_corrected else "amsgrad",
            mdw_mode="plain",
            mdw_matrices=[p in self.hidden for p in self.param_groups[0]["params"]],
        )
        self.physical_decay = PhysicalDecay(self.workspace)

    def load_state_dict(self, state_dict: StateDict) -> None:
        """Require matching adaptive histories before restoring device and rate storage."""
        group = state_dict["param_groups"][0]
        if any(
            group.get(key) != self.param_groups[0][key]
            for key in ("mdw_rule", "mdw_mode", "mdw_matrices")
        ):
            msg = "Recreate the same MDW rule, averaging mode and matrix selection before loading"
            raise ValueError(msg)
        super().load_state_dict(state_dict)

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        """Apply the MD proposal, decay the old physical weight, then repair storage."""
        group = self.param_groups[0]
        if not group["weight_decay"]:
            return super().step(closure)
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        parameters = group["params"]
        if any(p.grad is None or p.grad.is_sparse for p in parameters):
            msg = "MDW requires a dense gradient for every parameter"
            raise ValueError(msg)
        for parameter in parameters:
            self.workspace.like(parameter, "mdw.old").copy_(parameter)
        super().step()
        (auxiliary_rate,) = component_rates(self, "auxiliary_rate")
        for parameter in parameters:
            rate = group["lr"] if parameter in self.hidden else auxiliary_rate
            active = active_rates(self.workspace, parameter, "mdw.active", rate)
            corrected = self.physical_decay.apply(
                parameter, self.workspace.like(parameter, "mdw.old"), rate, group["weight_decay"]
            )
            copy_active(parameter, corrected, active=active)
            if parameter in self.hidden:
                state = self.state[parameter]
                self.physical_decay.repair(
                    parameter, (state["row"], state["column"]), state["radius"], active=active
                )
        return loss


class AdamMDW(AMSGradMDW):
    """Bias-corrected Adam on every factor, plus physical AdamW decay.

    Both moments follow native Adam's convention, including epsilon after
    the bias-corrected variance square root. No maximum buffers are stored.
    With matrix_params=[], auxiliary updates agree with native AdamW in
    exact arithmetic. Floating-point operation order can differ.
    Lean: Transformer.MDW.adamMDWProposal and adam_auxiliary_native.
    """

    _bias_corrected = True
    moment_update = staticmethod(adam)
    moment_buffers = ("m", "v")
    moments = "bias_corrected_Adam_factor_histories_without_running_maxima"
