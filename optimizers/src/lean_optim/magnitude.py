"""AMSGradMD: AMSGrad on the magnitude-direction factorization of the hidden matrices.

A port of `amsgrad_extensions_benchmark.optimizer.ExtensionOptimizer`, with
its arithmetic. A hidden matrix W is read as softplus(r)_i D_ij softplus(c)_j,
from raw gains r and c that start at softplus^-1(1) and the radius |W_0| of
the initial matrix. An update moves D, r and c by three AMSGrads, projects D
back onto the sphere of that radius and writes W fused from the three; every
other parameter moves by AMSGrad at the auxiliary rate. The optimizer writes
the new values instead of applying a direction, so it is no direction
optimizer and takes no stage, and its state is created with it: the radius
is the initial matrix's. The state of a parameter is flat, its gains and the
buffers of each AMSGrad under a prefix, so checkpoints and graph warm-ups
carry it like any other.

Under the guard, `Transformer.MagnitudeDirection.checkedProposal`, the whole
proposal is accepted when its displacement over the rate passes the descent
guard, every value is finite and no matrix is zero; otherwise every parameter
takes a gradient step at the rate, halved for a matrix it would zero, and the
row gains are rebalanced so that the stored direction keeps the radius
(`rebalanceStorage`, the identity on an accepted proposal in exact
arithmetic, which the benchmark skipped).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from torch import Tensor
from torch.nn import functional

from ._rates import active_rates, component_rates, copy_active
from ._workspace import BufferedOptimizer, Workspace, descent_test, joint_metrics

if TYPE_CHECKING:
    from collections.abc import Callable

    from ._types import (
        Closure,
        Loss,
        MagnitudeSpec,
        ParameterModel,
        Report,
        Scalar,
        StateDict,
        TensorState,
    )

BUFFERS = ("m", "v", "maximum")


def amsgrad(  # noqa: PLR0913, PLR0917 - Preserve the established optimizer/recurrence signature.
    x: Tensor,
    gradient: Tensor,
    state: TensorState,
    prefix: str,
    rate: Scalar,
    beta: float,
    beta2: float,
    eps: float,
    workspace: Workspace | None = None,
    *,
    output: Tensor | None = None,
    denominator: Tensor | None = None,
) -> Tensor:
    """Update buffers and x, following MagnitudeDirection.amsgradVectorCallback."""
    m, v, maximum = (state[prefix + name] for name in BUFFERS)
    m.mul_(beta).add_(gradient, alpha=1 - beta)
    v.mul_(beta2).addcmul_(gradient, gradient, value=1 - beta2)
    torch.maximum(maximum, v, out=maximum)
    if workspace is None:
        return x - rate * m / (eps + maximum.sqrt())
    if denominator is None:
        denominator = workspace.like(x, "amsgrad.denominator")
    torch.sqrt(maximum, out=denominator)
    denominator.add_(eps)
    if output is None:
        output = workspace.like(x, "amsgrad.output")
    torch.mul(m, rate, out=output)
    output.div_(denominator)
    return torch.sub(x, output, out=output)


def factored_state(weight: Tensor, buffers: tuple[str, ...] = BUFFERS) -> TensorState:
    """Initialize unit gains, the matrix radius and zero AMSGrad buffers."""
    radius = torch.linalg.vector_norm(weight.detach()).clone()
    if not bool(torch.isfinite(radius) & (radius > 0)):
        msg = "AMSGradMD factors finite nonzero matrices"
        raise ValueError(msg)
    gain = math.log(math.expm1(1.0))
    rows, columns = weight.shape
    state = {
        "row": weight.new_full((rows,), gain),
        "column": weight.new_full((columns,), gain),
        "radius": radius,
    }
    for prefix, like in (("", weight), ("row.", state["row"]), ("column.", state["column"])):
        state.update({prefix + name: torch.zeros_like(like) for name in buffers})
    return state


def proposal(  # noqa: PLR0913, PLR0917 - Preserve the established optimizer/recurrence signature.
    weight: Tensor,
    state: TensorState,
    gradient: Tensor,
    direction_rate: Scalar,
    gain_rate: Scalar,
    beta: float,
    beta2: float,
    eps: float,
    workspace: Workspace | None = None,
    *,
    update: Callable[..., Tensor] = amsgrad,
) -> tuple[Tensor, Tensor, Tensor]:
    """Compute amsgradMDProposal from gradients evaluated at the old factors."""
    if workspace is not None:
        return buffered_proposal(
            weight,
            state,
            gradient,
            direction_rate,
            gain_rate,
            beta,
            beta2,
            eps,
            workspace,
            update=update,
        )
    row, column = state["row"], state["column"]
    gains_row, gains_column = functional.softplus(row), functional.softplus(column)
    direction = weight / (gains_row[:, None] * gains_column[None, :])
    direction_gradient = gradient * gains_row[:, None] * gains_column[None, :]
    row_gradient = (direction * gradient * gains_column[None, :]).sum(1) * row.sigmoid()
    column_gradient = (direction * gradient * gains_row[:, None]).sum(0) * column.sigmoid()
    candidate = update(direction, direction_gradient, state, "", direction_rate, beta, beta2, eps)
    next_row = update(row, row_gradient, state, "row.", gain_rate, beta, beta2, eps)
    next_column = update(column, column_gradient, state, "column.", gain_rate, beta, beta2, eps)
    norm = torch.linalg.vector_norm(candidate)
    # The projection of a zero candidate is zero, as in `matrixProject`; the guard rejects it.
    scale = torch.where(norm > 0, state["radius"] / torch.where(norm > 0, norm, 1.0), 0.0)
    fused = (
        candidate
        * scale
        * functional.softplus(next_row)[:, None]
        * functional.softplus(next_column)[None, :]
    )
    return fused, next_row, next_column


def inverse_softplus(gain: Tensor) -> Tensor:
    """log(exp(gain) - 1), without overflow or cancellation."""
    return gain + torch.log(-torch.expm1(-gain))


def rebalance(weight: Tensor, row: Tensor, column: Tensor, radius: Tensor) -> Tensor:
    """Compute rebalanceStorage row gains for the stored direction radius."""
    direction = weight / (functional.softplus(row)[:, None] * functional.softplus(column)[None, :])
    return inverse_softplus(torch.linalg.vector_norm(direction) / radius * functional.softplus(row))


def softplus(value: Tensor, workspace: Workspace) -> Tensor:
    # functional.softplus has no out keyword; its ATen out overload is the same kernel.
    """Evaluate the standard softplus kernel into a reusable output."""
    return torch.ops.aten.softplus.out(value, out=workspace.like(value, "softplus"))


def buffered_proposal(  # noqa: PLR0913, PLR0917 - Preserve the established optimizer/recurrence signature.
    weight: Tensor,
    state: TensorState,
    gradient: Tensor,
    direction_rate: Scalar,
    gain_rate: Scalar,
    beta: float,
    beta2: float,
    eps: float,
    workspace: Workspace,
    *,
    update: Callable[..., Tensor] = amsgrad,
) -> tuple[Tensor, Tensor, Tensor]:
    """Compute the factorized AMSGrad candidate in reusable work buffers."""
    row, column = state["row"], state["column"]
    gains_row, gains_column = softplus(row, workspace), softplus(column, workspace)
    direction = workspace.like(weight, "proposal.direction")
    torch.mul(gains_row[:, None], gains_column[None, :], out=direction)
    torch.div(weight, direction, out=direction)
    direction_gradient = torch.mul(
        gradient, gains_row[:, None], out=workspace.like(weight, "proposal.gradient")
    )
    direction_gradient.mul_(gains_column[None, :])
    work = torch.mul(direction, gradient, out=workspace.like(weight, "proposal.work"))
    work.mul_(gains_column[None, :])
    row_gradient = torch.sum(work, dim=1, out=workspace.like(row, "gradient"))
    row_gradient.mul_(torch.sigmoid(row, out=workspace.like(row, "sigmoid")))
    torch.mul(direction, gradient, out=work)
    work.mul_(gains_row[:, None])
    column_gradient = torch.sum(work, dim=0, out=workspace.like(column, "gradient"))
    column_gradient.mul_(torch.sigmoid(column, out=workspace.like(column, "sigmoid")))
    candidate = update(
        direction,
        direction_gradient,
        state,
        "",
        direction_rate,
        beta,
        beta2,
        eps,
        workspace,
        output=work,
        denominator=direction_gradient,
    )
    next_row = update(row, row_gradient, state, "row.", gain_rate, beta, beta2, eps, workspace)
    next_column = update(
        column, column_gradient, state, "column.", gain_rate, beta, beta2, eps, workspace
    )
    norm = workspace.norm(candidate)
    positive = workspace.tensor(norm, "projection.positive", dtype=torch.bool)
    torch.gt(norm, 0, out=positive)
    scale = torch.div(
        state["radius"], workspace.safe(norm), out=workspace.like(norm, "projection.scale")
    )
    zero = workspace.tensor(norm, "projection.zero", ()).zero_()
    torch.where(positive, scale, zero, out=scale)
    candidate.mul_(scale).mul_(softplus(next_row, workspace)[:, None]).mul_(
        softplus(next_column, workspace)[None, :]
    )
    return candidate, next_row, next_column


def buffered_rebalance(  # noqa: PLR0913 - Keep the factorization arguments explicit.
    weight: Tensor,
    row: Tensor,
    column: Tensor,
    radius: Tensor,
    workspace: Workspace,
    *,
    reference: Tensor,
) -> Tensor:
    """Rebalance raw row gains while retaining the stored direction radius."""
    direction = workspace.like(reference, "proposal.direction")
    torch.mul(
        softplus(row, workspace)[:, None], softplus(column, workspace)[None, :], out=direction
    )
    torch.div(weight, direction, out=direction)
    gain = torch.mul(
        softplus(row, workspace),
        workspace.norm(direction).div_(radius),
        out=workspace.like(row, "rebalance.gain"),
    )
    work = torch.neg(gain, out=workspace.like(row, "rebalance.work"))
    work.expm1_().neg_().log_()
    return gain.add_(work)


class AMSGradMD(BufferedOptimizer):
    """The `amsgradmd` rule; given the guard's `sigma`, `amsgradmd_guarded`.

    `rate` is the rate the schedule sets: a float, or the one-element device
    tensor of the capturable form, read when an update runs. The direction
    moves at it unless the spec gives the guarded variant its own direction
    rate; the gains and the other parameters have independent base rates.
    An attached PyTorch scheduler scales all those rates by lr / initial_lr.
    """

    moments = "raw_AMSGrad_buffers_of_directions_and_auxiliary_parameters_for_AMSGradMD"
    moment_update = staticmethod(amsgrad)
    moment_buffers = BUFFERS

    def add_param_group(self, group: StateDict) -> None:
        """Register the supported parameter group."""
        if self.param_groups:
            msg = "AMSGradMD supports one joint parameter group"
            raise ValueError(msg)
        super().add_param_group(group)

    def __init__(
        self,
        spec: MagnitudeSpec,
        model: ParameterModel,
        rate: Scalar | None = None,
        sigma: float | None = None,
    ) -> None:
        """Initialize the parameters, hyperparameters and persistent state."""
        if sigma is None and spec.direction_rate is not None:
            msg = (
                "Unguarded, AMSGradMD moves directions at its rate; "
                "direction_rate belongs to the guard"
            )
            raise ValueError(msg)
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        super().__init__(parameters, {"lr": 0.0 if rate is None else rate})
        self.spec, self.sigma = spec, sigma
        self.names = {parameter: name for name, parameter in model.named_parameters()}
        self.hidden = set(model.hidden_matrices())
        for parameter in parameters:
            self.state[parameter].update(
                factored_state(parameter, self.moment_buffers)
                if parameter in self.hidden
                else {name: torch.zeros_like(parameter) for name in self.moment_buffers}
            )
        if sigma is not None:
            self.state["guard"] = {
                key: torch.zeros((), dtype=torch.int64, device=parameters[0].device)
                for key in ("updates", "accepted", "halved")
            }

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        """Apply one update, evaluating an optional closure with gradients enabled."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        spec, rate, parameters = (
            self.spec,
            self.param_groups[0]["lr"],
            self.param_groups[0]["params"],
        )
        if any(parameter.grad is None for parameter in parameters):
            msg = "AMSGradMD requires a gradient for every parameter"
            raise ValueError(msg)
        if hasattr(self, "_clip_norm"):
            torch.nn.utils.clip_grad_norm_(parameters, self._clip_norm)
        beta, beta2 = spec.betas
        direction_rate, gain_rate, auxiliary_rate = component_rates(
            self, "direction_rate", "gain_rate", "auxiliary_rate"
        )
        direction_rate = rate if direction_rate is None else direction_rate
        fallback_rates = () if self.sigma is None else (rate,)
        matrix_active = active_rates(
            self.workspace,
            parameters[0],
            "matrix.active",
            direction_rate,
            gain_rate,
            *fallback_rates,
        )
        auxiliary_active = active_rates(
            self.workspace, parameters[0], "auxiliary.active", auxiliary_rate, *fallback_rates
        )
        candidates, gains = [], {}
        for parameter in parameters:
            state, gradient = self.state[parameter], parameter.grad
            if parameter in self.hidden:
                candidate, *gains[parameter] = proposal(
                    parameter,
                    state,
                    gradient,
                    direction_rate,
                    gain_rate,
                    beta,
                    beta2,
                    spec.eps,
                    self.workspace,
                    update=self.moment_update,
                )
            else:
                candidate = self.moment_update(
                    parameter,
                    gradient,
                    state,
                    "",
                    auxiliary_rate,
                    beta,
                    beta2,
                    spec.eps,
                    self.workspace,
                    output=self.workspace.like(parameter, "proposal.work"),
                    denominator=self.workspace.like(parameter, "proposal.gradient"),
                )
            candidates.append(candidate)
        accepted = None if self.sigma is None else self.check(parameters, candidates, rate)
        for parameter, candidate in zip(parameters, candidates, strict=True):
            if parameter in self.hidden:
                state, (row, column) = self.state[parameter], gains[parameter]
                if accepted is not None:
                    balanced = buffered_rebalance(
                        candidate, row, column, state["radius"], self.workspace, reference=parameter
                    )
                    torch.where(accepted, row, balanced, out=row)
                copy_active(state["row"], row, active=matrix_active)
                copy_active(state["column"], column, active=matrix_active)
            copy_active(
                parameter,
                candidate,
                active=matrix_active if parameter in self.hidden else auxiliary_active,
            )
        return loss

    def check(self, parameters: list[Tensor], candidates: list[Tensor], rate: Scalar) -> Tensor:
        """Check and replace candidates in place; return whether the proposal was accepted."""
        gradients = [parameter.grad for parameter in parameters]
        directions = [
            torch.sub(p, c, out=self.workspace.like(p, "proposal.direction")).div_(rate)
            for p, c in zip(parameters, candidates, strict=True)
        ]
        accepted = descent_test(
            self.workspace,
            joint_metrics(self.workspace, parameters, directions, work_name="proposal.gradient"),
            self.sigma,
        )
        for parameter, candidate in zip(parameters, candidates, strict=True):
            mask = self.workspace.tensor(parameter, "guard.mask", dtype=torch.bool)
            work = torch.abs(candidate, out=self.workspace.like(parameter, "proposal.gradient"))
            torch.lt(work, math.inf, out=mask)
            finite = self.workspace.tensor(parameter, "guard.finite", (), torch.bool)
            torch.all(mask, out=finite)
            torch.logical_and(accepted, finite, out=accepted)
            if parameter in self.hidden:
                torch.ne(candidate, 0, out=mask)
                torch.any(mask, out=finite)
                torch.logical_and(accepted, finite, out=accepted)
        guard = self.state["guard"]
        rejected = torch.logical_not(accepted, out=self.workspace.like(accepted, "rejected"))
        for index, (parameter, gradient) in enumerate(zip(parameters, gradients, strict=False)):
            work = torch.mul(
                gradient, rate, out=self.workspace.like(parameter, "proposal.gradient")
            )
            torch.sub(parameter, work, out=work)
            if parameter in self.hidden:
                mask = self.workspace.tensor(parameter, "guard.mask", dtype=torch.bool)
                torch.eq(work, 0, out=mask)
                zeroed = torch.all(
                    mask, out=self.workspace.tensor(parameter, "guard.zero", (), torch.bool)
                )
            else:
                zeroed = self.workspace.tensor(parameter, "guard.zero", (), torch.bool).zero_()
            fallback = torch.mul(
                gradient, 0.5, out=self.workspace.like(parameter, "proposal.direction")
            )
            torch.where(zeroed, fallback, gradient, out=fallback)
            fallback.mul_(rate)
            torch.sub(parameter, fallback, out=fallback)
            torch.where(accepted, candidates[index], fallback, out=candidates[index])
            torch.logical_and(zeroed, rejected, out=zeroed)
            guard["halved"].add_(zeroed)
        guard["updates"].add_(1)
        guard["accepted"].add_(accepted)
        return accepted

    def report(self) -> Report:
        """Per hidden matrix, its radius and the factorization it ends with; the guard's counts."""
        matrices = {}
        for parameter in self.param_groups[0]["params"]:
            if parameter not in self.hidden:
                continue
            state = self.state[parameter]
            row, column = functional.softplus(state["row"]), functional.softplus(state["column"])
            norm, radius = (
                (parameter / (row[:, None] * column[None, :])).norm().item(),
                state["radius"].item(),
            )
            matrices[self.names[parameter]] = {
                "radius": radius,
                "recovered_norm": norm,
                "relative_sphere_error": abs(norm - radius) / radius,
                "row_gain_min": row.min().item(),
                "row_gain_max": row.max().item(),
                "col_gain_min": column.min().item(),
                "col_gain_max": column.max().item(),
            }
        report = {"magnitude": matrices}
        if self.sigma is not None:
            updates, accepted, halved = (
                self.state["guard"][key].item() for key in ("updates", "accepted", "halved")
            )
            report["guard"] = {
                "updates": updates,
                "accepted": accepted,
                "acceptance": accepted / max(updates, 1),
                "halved": halved,
            }
        return report
