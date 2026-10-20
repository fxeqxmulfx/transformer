"""Matrix rules: Muon's orthogonalized momentum and DASH's blocked Shampoo.

Ports of `optimizer_benchmark.matrix.MuonOptimizer` and of the batched DASH
step of `full_compile_benchmark.optimizer`, with their arithmetic. Muon's
hidden matrices are the model's: the weights of its blocks, which for GPTMini
are the matrices the benchmark found by name.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from . import config as optimizers
from ._parameters import MATRIX_NDIM
from .coordinate import bias_correction
from .direction import DirectionOptimizer
from .roots import build_root

if TYPE_CHECKING:
    from ._types import DashSpec, MuonSpec, ParameterModel, Scalar, StateDict, TensorState
    from ._workspace import Workspace


def polar(matrix: Tensor, steps: int = 5) -> Tensor:
    """Newton-Schulz steps toward the polar factor, from the Frobenius-normalized matrix.

    A tall matrix is iterated transposed, which the odd polynomial commutes with.
    """
    transpose = matrix.size(0) > matrix.size(1)
    x = matrix.T if transpose else matrix
    norm = x.square().sum().sqrt()
    x = x / torch.where(norm > 0, norm, torch.ones_like(norm))
    for _ in range(steps):
        gram = x @ x.T
        x = 3.4445 * x + (-4.775 * gram + 2.0315 * (gram @ gram)) @ x
    return x.T if transpose else x


def buffered_polar(matrix: Tensor, workspace: Workspace, steps: int = 5) -> Tensor:
    """Consume a work matrix to evaluate the polar polynomial without allocations."""
    transpose = matrix.size(0) > matrix.size(1)
    x = workspace.like(matrix, "polar.x")
    # The consumed Nesterov input becomes the next iterate after x is copied.
    next_x = matrix
    squared = next_x
    x.copy_(matrix)
    if transpose:
        x, next_x, squared = x.T, next_x.T, squared.T
    torch.square(x, out=squared)
    norm = workspace.sum(squared, reference=matrix, name="polar.norm").sqrt_()
    x.div_(workspace.safe(norm))
    size = x.size(0)
    gram = workspace.tensor(matrix, "polar.gram", (size, size))
    polynomial = workspace.tensor(matrix, "polar.polynomial", (size, size))
    product = workspace.tensor(matrix, "polar.product", x.shape)
    for _ in range(steps):
        torch.matmul(x, x.T, out=gram)
        torch.matmul(gram, gram, out=polynomial)
        gram.mul_(-4.775)
        polynomial.mul_(2.0315)
        torch.add(gram, polynomial, out=polynomial)
        torch.matmul(polynomial, x, out=product)
        torch.mul(x, 3.4445, out=next_x)
        next_x.add_(product)
        x, next_x = next_x, x
    return x.T if transpose else x


def muon_direction(
    parameter: Tensor,
    state: StateDict,
    spec: MuonSpec,
    workspace: Workspace,
    *,
    matrix_update: bool,
) -> Tensor:
    """Native Muon/auxiliary Adam direction, reusable by Schedule-Free.

    Source: arXiv:2502.16982, Sections 2.1-2.2; the original operation order
    is retained. Matrix membership is supplied explicitly, so a wrapper can
    restore it with its state. Extra averaging buffers do not change moments.
    """
    gradient = parameter.grad
    if matrix_update:
        if "momentum" not in state:
            state["momentum"] = torch.zeros_like(parameter)
        moment = state["momentum"]
        moment.mul_(spec.momentum).add_(gradient)
        nesterov = workspace.like(parameter, "nesterov")
        torch.mul(moment, spec.momentum, out=nesterov)
        nesterov.add_(gradient)
        return buffered_polar(nesterov, workspace).mul_(0.2 * math.sqrt(max(parameter.shape)))
    if "m" not in state:
        state.update(
            m=torch.zeros_like(parameter),
            v=torch.zeros_like(parameter),
            step=parameter.new_zeros((), dtype=torch.float64),
        )
    (beta, beta2), m, v, step = (0.9, 0.999), state["m"], state["v"], state["step"].add_(1)
    m.mul_(beta).add_(gradient, alpha=1 - beta)
    v.mul_(beta2).addcmul_(gradient, gradient, value=1 - beta2)
    direction = workspace.like(parameter, "direction")
    numerator = workspace.like(parameter, "numerator")
    torch.div(m, bias_correction(beta, step, workspace, "bias1"), out=numerator)
    torch.div(v, bias_correction(beta2, step, workspace, "bias2"), out=direction)
    direction.sqrt_().add_(1e-8)
    torch.div(numerator, direction, out=direction)
    return direction.mul_(spec.auxiliary)


class Muon(DirectionOptimizer):
    """Apply Muon to selected matrices and bias-corrected auxiliary Adam elsewhere."""

    moments = (
        "raw_momentum_of_hidden_matrices_and_auxiliary_Adam_moments_before_bias_correction_for_Muon"
    )

    def __init__(self, spec: MuonSpec, model: ParameterModel, rate: Scalar | None = None) -> None:
        """Initialize the parameters, hyperparameters and persistent state."""
        super().__init__(model, rate)
        self.spec, self.hidden = spec, set(model.hidden_matrices())

    def first_moment(self, parameter: Tensor) -> str | None:
        """Return the raw first-moment state key, if the rule keeps one."""
        return "momentum" if parameter in self.hidden else "m"

    def direction(self, parameter: Tensor, state: TensorState) -> Tensor:
        """Compute the update direction from the current gradient and state."""
        return muon_direction(
            parameter, state, self.spec, self.workspace, matrix_update=parameter in self.hidden
        )


class Dash(DirectionOptimizer):
    """The `dash_*` rules: one batched history per block shape, in the stage-like state "dash".

    The blocks of every parameter, a vector as one column, are gathered from
    the concatenated gradients by position, shape by shape, in parameter
    order; the directions are scattered back the same way.
    """

    buffers = ("left", "right", "moment", "accumulator")

    def load_state_dict(self, state_dict: StateDict) -> None:
        """Restore mathematical state and discard temporary work buffers."""
        if state_dict["param_groups"][0].get("block", self.spec.block) != self.spec.block:
            msg = "Recreate Dash with the checkpoint's block size"
            raise ValueError(msg)
        super().load_state_dict(state_dict)
        self.root = build_root(self.spec.solver)

    def __init__(self, spec: DashSpec, model: ParameterModel, rate: Scalar | None = None) -> None:
        """Initialize the parameters, hyperparameters and persistent state."""
        super().__init__(model, rate)
        if self.capturable and isinstance(spec.solver, optimizers.EVD):
            msg = "torch cannot capture an eigendecomposition: run Dash with EVD under Eager()"
            raise ValueError(msg)
        self.spec, self.root = spec, build_root(spec.solver)
        self.packed_dtype = self.param_groups[0]["params"][0].dtype
        blocks, offset = {}, 0
        for parameter in self.param_groups[0]["params"]:
            self.packed_dtype = torch.promote_types(self.packed_dtype, parameter.dtype)
            rows, columns = (
                parameter.shape if parameter.ndim == MATRIX_NDIM else (parameter.numel(), 1)
            )
            location = (
                torch.arange(parameter.numel(), device=parameter.device).reshape(rows, columns)
                + offset
            )
            offset += parameter.numel()
            for i in range(0, rows, spec.block):
                for j in range(0, columns, spec.block):
                    block = location[i : i + spec.block, j : j + spec.block]
                    blocks.setdefault(tuple(block.shape), []).append(block)
        self.positions = {shape: torch.stack(group) for shape, group in blocks.items()}

        def zeros(*shape: int) -> Tensor:
            return parameter.new_zeros(shape, dtype=self.packed_dtype)

        self.state["dash"] = {
            f"{height}x{width}.{name}": zeros(len(positions), *shape)
            for (height, width), positions in self.positions.items()
            for name, shape in zip(
                self.buffers,
                ((height, height), (width, width), (height, width), (height, width)),
                strict=False,
            )
        }

    def directions(self, parameters: list[Tensor]) -> list[Tensor]:
        """Compute every direction before moving any parameter."""
        if len(parameters) != len(self.param_groups[0]["params"]):
            msg = "Dash preconditions every parameter, so each needs a gradient"
            raise ValueError(msg)
        spec, state = self.spec, self.state["dash"]
        (beta, graft_beta, graft_beta2), eps = (spec.beta, *spec.graft_betas), spec.eps
        packed = self.workspace.tensor(
            parameters[0], "packed", (sum(p.numel() for p in parameters),), dtype=self.packed_dtype
        )
        directions = self.workspace.like(packed, "directions")
        offset = 0
        for parameter in parameters:
            packed[offset : offset + parameter.numel()].view_as(parameter).copy_(parameter.grad)
            offset += parameter.numel()
        for (height, width), positions in self.positions.items():
            left, right, moment, accumulator = (
                state[f"{height}x{width}.{name}"] for name in self.buffers
            )
            gradient = self.workspace.like(moment, "gradient")
            torch.take(packed, positions, out=gradient)
            transpose = gradient.transpose(-2, -1)
            left_product = self.workspace.like(left, "gram")
            right_product = self.workspace.like(right, "gram")
            torch.matmul(gradient, transpose, out=left_product)
            torch.matmul(transpose, gradient, out=right_product)
            left.mul_(beta).add_(left_product, alpha=1 - beta)
            right.mul_(beta).add_(right_product, alpha=1 - beta)
            moment.mul_(graft_beta).add_(gradient, alpha=1 - graft_beta)
            accumulator.mul_(graft_beta2).addcmul_(gradient, gradient, value=1 - graft_beta2)
            reference = torch.sqrt(accumulator, out=self.workspace.like(moment, "reference"))
            reference.add_(eps)
            torch.div(moment, reference, out=reference)
            work = self.workspace.like(moment, "work")
            output = self.workspace.like(moment, "output")
            torch.matmul(self.root(left, eps), gradient, out=work)
            torch.matmul(work, self.root(right, eps), out=output)
            norm = self.workspace.tensor(moment, "norm", moment.shape[:-2])
            reference_norm = self.workspace.tensor(moment, "reference_norm", moment.shape[:-2])
            reference.square_()
            torch.sum(reference, dim=(-2, -1), out=reference_norm)
            reference_norm.sqrt_()
            torch.square(output, out=reference)
            torch.sum(reference, dim=(-2, -1), out=norm)
            norm.sqrt_()
            # Lean's total division gives zero at a zero Shampoo direction.
            reference_norm.div_(self.workspace.safe(norm))
            output.mul_(reference_norm[..., None, None])
            directions.scatter_(0, positions.flatten(), output.flatten())
        offset, result = 0, []
        for parameter in parameters:
            result.append(directions[offset : offset + parameter.numel()].reshape_as(parameter))
            offset += parameter.numel()
        return result
