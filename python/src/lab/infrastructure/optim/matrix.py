"""Matrix rules: Muon's orthogonalized momentum and DASH's blocked Shampoo.

Ports of `optimizer_benchmark.matrix.MuonOptimizer` and of the batched DASH
step of `full_compile_benchmark.optimizer`, with their arithmetic. Muon's
hidden matrices are the model's: the weights of its blocks, which for GPTMini
are the matrices the benchmark found by name.
"""

import math

import torch

from ...domain import optimizers
from .direction import DirectionOptimizer
from .roots import build_root


def polar(matrix, steps=5):
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


class Muon(DirectionOptimizer):
    """The `muon` rule: Muon on the hidden matrices, bias-corrected Adam at a fraction of the rate elsewhere."""
    moments = "raw_momentum_of_hidden_matrices_and_auxiliary_Adam_moments_before_bias_correction_for_Muon"

    def __init__(self, spec, model, rate=None):
        super().__init__(model, rate)
        self.spec, self.hidden = spec, set(model.hidden_matrices())

    def first_moment(self, parameter):
        return "momentum" if parameter in self.hidden else "m"

    def direction(self, parameter, state):
        spec, gradient = self.spec, parameter.grad
        if parameter in self.hidden:
            if not state:
                state["momentum"] = torch.zeros_like(parameter)
            moment = state["momentum"]
            moment.mul_(spec.momentum).add_(gradient)
            nesterov = spec.momentum * moment + gradient
            return 0.2 * math.sqrt(max(parameter.shape)) * polar(nesterov)
        if not state:
            state.update(m=torch.zeros_like(parameter), v=torch.zeros_like(parameter),
                         step=parameter.new_zeros((), dtype=torch.float64))
        (beta, beta2), m, v, step = (0.9, 0.999), state["m"], state["v"], state["step"].add_(1)
        m.mul_(beta).add_(gradient, alpha=1 - beta)
        v.mul_(beta2).addcmul_(gradient, gradient, value=1 - beta2)
        return spec.auxiliary * ((m / (1 - beta ** step)) / ((v / (1 - beta2 ** step)).sqrt() + 1e-8))


class Dash(DirectionOptimizer):
    """The `dash_*` rules: one batched history per block shape, in the stage-like state "dash".

    The blocks of every parameter, a vector as one column, are gathered from
    the concatenated gradients by position, shape by shape, in parameter
    order; the directions are scattered back the same way.
    """
    buffers = ("left", "right", "moment", "accumulator")

    def __init__(self, spec, model, rate=None):
        super().__init__(model, rate)
        if rate is not None and isinstance(spec.solver, optimizers.EVD):
            raise ValueError("torch cannot capture an eigendecomposition: run Dash with EVD under Eager()")
        self.spec, self.root = spec, build_root(spec.solver)
        blocks, offset = {}, 0
        for parameter in self.param_groups[0]["params"]:
            rows, columns = parameter.shape if parameter.ndim == 2 else (parameter.numel(), 1)
            location = torch.arange(parameter.numel(), device=parameter.device).reshape(rows, columns) + offset
            offset += parameter.numel()
            for i in range(0, rows, spec.block):
                for j in range(0, columns, spec.block):
                    block = location[i:i + spec.block, j:j + spec.block]
                    blocks.setdefault(tuple(block.shape), []).append(block)
        self.positions = {shape: torch.stack(group) for shape, group in blocks.items()}
        zeros = lambda *shape: parameter.new_zeros(shape)
        self.state["dash"] = {
            f"{height}x{width}.{name}": zeros(len(positions), *shape)
            for (height, width), positions in self.positions.items()
            for name, shape in zip(self.buffers, ((height, height), (width, width), (height, width), (height, width)))}

    def directions(self, parameters):
        if len(parameters) != len(self.param_groups[0]["params"]):
            raise ValueError("Dash preconditions every parameter, so each needs a gradient")
        spec, state = self.spec, self.state["dash"]
        (beta, graft_beta, graft_beta2), eps = (spec.beta, *spec.graft_betas), spec.eps
        packed = torch.cat([parameter.grad.reshape(-1) for parameter in parameters])
        directions = torch.zeros_like(packed)
        for (height, width), positions in self.positions.items():
            left, right, moment, accumulator = (state[f"{height}x{width}.{name}"] for name in self.buffers)
            gradient = packed[positions]
            transpose = gradient.transpose(-2, -1)
            left.mul_(beta).add_(gradient @ transpose, alpha=1 - beta)
            right.mul_(beta).add_(transpose @ gradient, alpha=1 - beta)
            moment.mul_(graft_beta).add_(gradient, alpha=1 - graft_beta)
            accumulator.mul_(graft_beta2).addcmul_(gradient, gradient, value=1 - graft_beta2)
            reference = moment / (accumulator.sqrt() + eps)
            output = self.root(left, eps) @ gradient @ self.root(right, eps)
            norm = output.square().sum((-2, -1)).sqrt()
            reference_norm = reference.square().sum((-2, -1)).sqrt()
            # Lean's total division gives zero at a zero Shampoo direction.
            output = output * (reference_norm / torch.where(norm > 0, norm, torch.ones_like(norm)))[..., None, None]
            directions.scatter_(0, positions.flatten(), output.flatten())
        offset, result = 0, []
        for parameter in parameters:
            result.append(directions[offset:offset + parameter.numel()].reshape_as(parameter))
            offset += parameter.numel()
        return result
