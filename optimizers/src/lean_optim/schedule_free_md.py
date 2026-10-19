"""Schedule-Free AMSGrad in magnitude/direction coordinates.

This new composition uses arXiv:2405.15682v4, Section 4.4, Algorithm 1,
arXiv:1904.03590v4, Algorithm 1 and arXiv:2606.25971v2, Section 3.1 and
Appendix A. The base direction is projected before averaging; averages
and queries are formed in direction/raw-gain coordinates. All three
chain-rule gradients use the old query factors. Each component has its
own squared-effective-rate total. Auxiliary tensors use the same
Schedule-Free AMSGrad rule. Smooth softplus is evaluated with logaddexp,
so its derivative is sigmoid throughout the raw-gain domain.

Lean: Transformer.ScheduleFree.AMSGrad.mdStep and mdQueryWeight.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from ._parameters import Parameters, settings
from ._rates import component_rates
from .schedule_free import ScheduleFreeAMSGradW

if TYPE_CHECKING:
    from collections.abc import Iterable

    from ._types import Closure, Loss, Params, StateDict


class ScheduleFreeAMSGradMD(ScheduleFreeAMSGradW):
    """Average latent direction and raw gains, then fuse the model weights.

    beta1 supplies Schedule-Free interpolation to all components. Selected
    matrices have independent direction, row and column histories and rates;
    other tensors use auxiliary_rate. A zero component rate freezes that
    component while its gradient history continues to advance. eval() fuses
    the averaged factors; train() fuses their queries. Direction averages
    and queries may lie inside the initial-radius ball.
    """

    def __init__(  # noqa: PLR0913, PLR0917 - Match the matrix optimizer hyperparameter interface.
        self,
        params: Params,
        lr: float = 1e-3,
        gain_rate: float = 1e-3,
        auxiliary_rate: float = 3e-4,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        warmup: int = 0,
        *,
        matrix_params: Iterable[Tensor] | None = None,
    ) -> None:
        """Initialize unit gains and separate adaptive factor histories."""
        settings(gain_rate)
        settings(auxiliary_rate)
        model = Parameters(params, matrix_params)
        super().__init__(model.parameters(), lr, betas, 0.0, eps, warmup)
        self.param_groups[0].update(gain_rate=gain_rate, auxiliary_rate=auxiliary_rate)
        gain = math.log(math.expm1(1.0))
        for parameter in model.hidden_matrices():
            radius = torch.linalg.vector_norm(parameter.detach())
            if not bool(torch.isfinite(radius) & (radius > 0)):
                msg = "ScheduleFreeAMSGradMD factors finite nonzero matrices"
                raise ValueError(msg)
            state = self.state[parameter]
            state["radius"] = radius
            rows, columns = parameter.shape
            self.state[parameter] = {
                "direction": state,
                "row": self._factor_state(parameter.new_full((rows,), gain)),
                "column": self._factor_state(parameter.new_full((columns,), gain)),
            }
        self.train()

    def _factor_state(self, value: Tensor) -> StateDict:
        return {
            "base": value,
            "evaluation": value.clone(),
            "step": 0,
            **self._second_state(value),
        }

    def _point(self, state: StateDict, *, training: bool) -> Tensor:
        if not training:
            return state["evaluation"]
        query = self.workspace.like(state["base"], "query")
        self._write(query, state, training=True)
        return query

    def _gain(self, point: Tensor, state: StateDict) -> Tensor:
        reference = state["base"]
        zero = self.workspace.tensor(reference, "softplus.zero", ()).zero_()
        return torch.logaddexp(point, zero, out=self.workspace.like(reference, "softplus"))

    @torch.no_grad()
    def _mode(self, *, training: bool) -> ScheduleFreeAMSGradMD:
        group = self.param_groups[0]
        group["training"] = training
        for parameter in group["params"]:
            state = self.state[parameter]
            if "direction" not in state:
                self._write(parameter, state, training=training)
                continue
            direction = self._point(state["direction"], training=training)
            row = self._gain(self._point(state["row"], training=training), state["row"])
            column = self._gain(self._point(state["column"], training=training), state["column"])
            torch.mul(direction, row[:, None], out=parameter)
            parameter.mul_(column[None, :])
        return self

    def _factor_gradients(
        self, parameter: Tensor, state: StateDict
    ) -> tuple[tuple[Tensor, Tensor, Tensor], tuple[Tensor, Tensor, Tensor]]:
        direction = self._point(state["direction"], training=True)
        row = self._point(state["row"], training=True)
        column = self._point(state["column"], training=True)
        row_gain, column_gain = self._gain(row, state["row"]), self._gain(column, state["column"])
        work = torch.mul(
            direction, parameter.grad, out=self.workspace.like(parameter, "md.gradient.work")
        )
        work.mul_(column_gain[None, :])
        row_gradient = torch.sum(
            work, dim=1, out=self.workspace.like(state["row"]["base"], "gradient")
        )
        row_gradient.mul_(
            torch.sigmoid(row, out=self.workspace.like(state["row"]["base"], "sigmoid"))
        )
        torch.mul(direction, parameter.grad, out=work)
        work.mul_(row_gain[:, None])
        column_gradient = torch.sum(
            work, dim=0, out=self.workspace.like(state["column"]["base"], "gradient")
        )
        column_gradient.mul_(
            torch.sigmoid(column, out=self.workspace.like(state["column"]["base"], "sigmoid"))
        )
        gradient = torch.mul(
            parameter.grad, row_gain[:, None], out=self.workspace.like(parameter, "md.gradient")
        )
        gradient.mul_(column_gain[None, :])
        return (direction, row, column), (gradient, row_gradient, column_gradient)

    def _step_factors(self, parameter: Tensor, state: StateDict, gain_rate: float) -> None:
        (direction, row, column), (gradient, row_gradient, column_gradient) = (
            self._factor_gradients(parameter, state)
        )
        group = self.param_groups[0]
        self._advance(direction, gradient, state["direction"], group["lr"])
        self._advance(row, row_gradient, state["row"], gain_rate)
        self._advance(column, column_gradient, state["column"], gain_rate)

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        """Apply chain-rule gradients at the query to the latent base factors."""
        loss, parameters = self._gradients(closure)
        gain_rate, auxiliary_rate = component_rates(self, "gain_rate", "auxiliary_rate")
        for parameter in parameters:
            state = self.state[parameter]
            if "direction" in state:
                self._step_factors(parameter, state, gain_rate)
            else:
                self._advance(parameter, parameter.grad, state, auxiliary_rate)
        self.train()
        return loss
