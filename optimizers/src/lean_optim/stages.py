"""Stages between a rule's directions and the update.

A stage is called with the optimizer, the parameters and their directions,
and returns the directions to apply. Its state is created with it, in the
optimizer's `state` under the stage's name.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

from ._workspace import descent_test, joint_metrics

if TYPE_CHECKING:
    from ._types import MaskSpec, ParameterModel, Report, TensorState
    from .direction import DirectionOptimizer


class Guard:
    """`Transformer.Optimization.descentGuard` on the joint parameter vector.

    The arithmetic of `full_compile_benchmark.step.FullStep`: squared norms
    and the alignment summed over parameters in order, and the decision as a
    device tensor, so a captured update replays it.
    """

    name = "guard"

    def __init__(self, sigma: float, optimizer: DirectionOptimizer) -> None:
        """Initialize the parameters, hyperparameters and persistent state."""
        self.sigma = sigma
        device = optimizer.param_groups[0]["params"][0].device
        optimizer.state[self.name] = {
            key: torch.zeros((), dtype=torch.int64, device=device)
            for key in ("updates", "accepted")
        }

    def __call__(
        self, optimizer: DirectionOptimizer, parameters: list[Tensor], directions: list[Tensor]
    ) -> list[Tensor]:
        """Transform the proposed directions before parameters move."""
        state = optimizer.state[self.name]
        accepted = descent_test(
            optimizer.workspace,
            joint_metrics(optimizer.workspace, parameters, directions),
            self.sigma,
        )
        state["updates"].add_(1)
        state["accepted"].add_(accepted)
        for parameter, direction in zip(parameters, directions, strict=True):
            torch.where(accepted, direction, parameter.grad, out=direction)
        return directions

    @staticmethod
    def report(state: TensorState) -> Report:
        """Return the recorded optimizer diagnostics."""
        updates, accepted = state["updates"].item(), state["accepted"].item()
        return {"updates": updates, "accepted": accepted, "acceptance": accepted / max(updates, 1)}


def cosine(moment: Tensor, gradient: Tensor, work: Tensor | None = None) -> Tensor:
    """Evaluate Magma.cosine with zero division and clamp roundoff to [-1, 1]."""
    numerator = (
        (moment * gradient).sum() if work is None else torch.mul(moment, gradient, out=work).sum()
    )
    denominator = moment.norm() * gradient.norm()
    return (
        numerator / torch.where(denominator > 0, denominator, torch.ones_like(denominator))
    ).clamp(-1, 1)


class Magma:
    """MAGMA's damped random masks on the hidden matrices.

    The arithmetic of `full_compile_benchmark.optimizer.propose`. The masks of
    the whole run are drawn at construction, a row per update and a column per
    hidden matrix in model order, as `full_compile_benchmark.step.mask_plan`
    drew them, and the row is chosen by a device counter, so a captured update
    replays it. Past the last row the plan starts over, which only a graph
    warm-up reaches. Each hidden matrix keeps its scale, its counts and, when
    the rule keeps no first moment of it, the moment the stage scores with.
    """

    name = "magma"

    def __init__(
        self,
        spec: MaskSpec,
        optimizer: DirectionOptimizer,
        model: ParameterModel,
        updates: int,
        seed: int,
    ) -> None:
        """Initialize the parameters, hyperparameters and persistent state."""
        self.tau = spec.tau
        names = {parameter: name for name, parameter in model.named_parameters()}
        self.masked = {parameter: names[parameter] for parameter in model.hidden_matrices()}
        self.columns = {parameter: column for column, parameter in enumerate(self.masked)}
        device = optimizer.param_groups[0]["params"][0].device
        generator = torch.Generator(device="cpu").manual_seed(seed)
        self.plan = (
            torch.rand((updates, len(self.masked)), generator=generator) < spec.survival
        ).to(device)
        state = {"updates": torch.zeros((), dtype=torch.int64, device=device)}
        for parameter, name in self.masked.items():
            state.update(
                {
                    f"{name}.scale": parameter.new_tensor(0.5),
                    f"{name}.scale_sum": parameter.new_zeros(()),
                    f"{name}.cosine_sum": parameter.new_zeros(()),
                    f"{name}.draws": torch.zeros((), dtype=torch.int64, device=device),
                    f"{name}.kept": torch.zeros((), dtype=torch.int64, device=device),
                }
            )
            if optimizer.first_moment(parameter) is None:
                state[f"{name}.moment"] = torch.zeros_like(parameter)
        optimizer.state[self.name] = state

    def __call__(
        self, optimizer: DirectionOptimizer, parameters: list[Tensor], directions: list[Tensor]
    ) -> list[Tensor]:
        """Transform the proposed directions before parameters move."""
        state = optimizer.state[self.name]
        index = optimizer.workspace.like(state["updates"], "mask.index")
        torch.remainder(state["updates"], len(self.plan), out=index)
        row = optimizer.workspace.tensor(self.plan, "mask.row", (1, self.plan.size(1)))
        masks = torch.index_select(self.plan, 0, index.reshape(1), out=row).squeeze(0)
        state["updates"].add_(1)
        result = []
        for parameter, direction in zip(parameters, directions, strict=True):
            name = self.masked.get(parameter)
            if name is None:
                result.append(direction)
                continue
            key = optimizer.first_moment(parameter)
            if key is None:
                moment = state[f"{name}.moment"].mul_(0.9).add_(parameter.grad, alpha=0.1)
            else:
                moment = optimizer.state[parameter][key]
            work = optimizer.workspace.like(parameter, "stage_work")
            numerator = optimizer.workspace.sum(torch.mul(moment, parameter.grad, out=work))
            moment_norm = optimizer.workspace.norm(moment)
            gradient_norm = optimizer.workspace.norm(
                parameter.grad, reference=parameter, name="gradient.norm"
            )
            denominator = torch.mul(
                moment_norm, gradient_norm, out=optimizer.workspace.like(numerator, "denominator")
            )
            alignment = torch.div(
                numerator, optimizer.workspace.safe(denominator), out=numerator
            ).clamp_(-1, 1)
            scale = state[f"{name}.scale"]
            mask = masks[self.columns[parameter]]
            damping = torch.div(
                alignment, self.tau, out=optimizer.workspace.like(alignment, "damping")
            )
            damping.sigmoid_()
            scale.mul_(0.9).add_(damping, alpha=0.1)
            state[f"{name}.draws"].add_(1)
            state[f"{name}.kept"].add_(mask)
            state[f"{name}.scale_sum"].add_(scale)
            state[f"{name}.cosine_sum"].add_(alignment)
            direction.mul_(scale)
            zero = optimizer.workspace.tensor(parameter, "stage_zero", ()).zero_()
            torch.where(mask, direction, zero, out=direction)
            result.append(direction)
        return result

    def report(self, state: TensorState) -> Report:
        """Per hidden matrix: its draws, the masks kept, and its scale on average and at the end."""
        blocks = {}
        for name in self.masked.values():
            draws, kept = state[f"{name}.draws"].item(), state[f"{name}.kept"].item()
            blocks[name] = {
                "draws": draws,
                "kept": kept,
                "survival_fraction": kept / draws if draws else None,
                "mean_damping": state[f"{name}.scale_sum"].item() / draws if draws else None,
                "mean_cosine": state[f"{name}.cosine_sum"].item() / draws if draws else None,
                "final_damping": state[f"{name}.scale"].item(),
            }
        draws = sum(block["draws"] for block in blocks.values())
        return {
            "blocks": blocks,
            "mask_draws": draws,
            "survival_fraction": sum(block["kept"] for block in blocks.values()) / draws
            if draws
            else None,
        }
