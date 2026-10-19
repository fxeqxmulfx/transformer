"""Schedule-Free physical MDW combinations, Transformer.MDW.freeMDStep.

Local extensions of arXiv:2405.15682v4, Section 4.4, Algorithm 1, and
arXiv:2606.25971v2, Appendix A. Old-query chain-rule gradients advance all
three latent bases. Query-point physical decay and storage repair precede
averaging. Neither the direction nor raw gains receive a separate penalty.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import Tensor

from ._mdw_decay import PhysicalDecay
from ._parameters import settings
from ._rates import active_rates
from .schedule_free_md import ScheduleFreeAMSGradMD

if TYPE_CHECKING:
    from collections.abc import Iterable

    from ._types import Params, StateDict


class ScheduleFreeAMSGradMDW(ScheduleFreeAMSGradMD):
    """Schedule-Free AMSGradMD with query-point decay of the physical base.

    Direction/raw-gain interpolation and independent squared-rate totals
    follow ScheduleFreeAMSGradMD. The fused base loses eta_direction * wd
    times the old fused query, then its direction and row gain are repaired
    before averaging. A zero adaptive gain rate can still allow the physical
    decay repair to move its base; its own averaging coefficient stays zero.
    Auxiliary tensors use ordinary Schedule-Free AMSGradW at their own rate.
    train()/eval() fuse the actual corrected factor histories. Eager only.
    """

    _use_maximum = True

    def __init__(  # noqa: PLR0913 - Match the MD family's established hyperparameter interface.
        self,
        params: Params,
        lr: float = 1e-3,
        *,
        gain_rate: float = 1e-3,
        auxiliary_rate: float = 3e-4,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        warmup: int = 0,
        weight_decay: float = 0.01,
        matrix_params: Iterable[Tensor] | None = None,
        rate: float | None = None,
    ) -> None:
        """Initialize the selected variance rule and zero per-factor latent penalties."""
        if isinstance(rate, Tensor):
            msg = "Schedule-Free MDW takes float rates and runs eagerly"
            raise TypeError(msg)
        settings(lr if rate is None else rate, weight_decay=weight_decay)
        super().__init__(
            params,
            lr if rate is None else rate,
            gain_rate,
            auxiliary_rate,
            betas,
            eps,
            warmup,
            matrix_params=matrix_params,
        )
        group = self.param_groups[0]
        group.update(
            weight_decay=weight_decay,
            mdw_rule="amsgrad" if self._use_maximum else "adam",
            mdw_mode="schedule_free",
            mdw_matrices=["direction" in self.state[p] for p in group["params"]],
        )
        for state in self.state.values():
            if "direction" in state:
                for name in ("direction", "row", "column"):
                    state[name]["weight_decay"] = 0.0
        self.physical_decay = PhysicalDecay(self.workspace, smooth=True)

    def load_state_dict(self, state_dict: StateDict) -> None:
        """Reject incompatible variance histories before restoring checkpoint placement."""
        group = state_dict["param_groups"][0]
        if any(
            group.get(key) != self.param_groups[0][key]
            for key in ("mdw_rule", "mdw_mode", "mdw_matrices")
        ):
            msg = "Recreate the same MDW rule, averaging mode and matrix selection before loading"
            raise ValueError(msg)
        super().load_state_dict(state_dict)

    def _step_factors(self, parameter: Tensor, state: StateDict, gain_rate: float) -> None:
        decay = self.param_groups[0]["weight_decay"]
        if not decay:
            super()._step_factors(parameter, state, gain_rate)
            return
        points, gradients = self._factor_gradients(parameter, state)
        states = tuple(state[name] for name in ("direction", "row", "column"))
        rates = (self.param_groups[0]["lr"], gain_rate, gain_rate)
        updates = [
            self._advance(point, gradient, factor, rate, average=False)
            for point, gradient, factor, rate in zip(points, gradients, states, rates, strict=True)
        ]
        direction, row, column = states
        fused = self.workspace.like(parameter, "mdw.base")
        fused.copy_(direction["base"]).mul_(self._gain(row["base"], row)[:, None])
        fused.mul_(self._gain(column["base"], column)[None, :])
        eta = updates[0][0]
        corrected = self.physical_decay.apply(fused, parameter, eta, decay)
        self.physical_decay.repair(
            corrected,
            (row["base"], column["base"]),
            direction["radius"],
            direction=direction["base"],
            active=active_rates(self.workspace, parameter, "mdw.active", eta),
        )
        for factor, (_, coefficient) in zip(states, updates, strict=True):
            self._average(factor, coefficient)


class ScheduleFreeAdamMDW(ScheduleFreeAMSGradMDW):
    """Schedule-Free MDW using raw variances instead of AMSGrad maxima.

    Beta1 controls interpolation, with no first-moment EMA. Effective rates,
    raw-variance epsilon placement and squared-rate averaging follow the
    paper's Schedule-Free AdamW. The physical correction follows freeMDStep
    with useMaximum=false. No maximum buffer is allocated or serialized.
    """

    _use_maximum = False
