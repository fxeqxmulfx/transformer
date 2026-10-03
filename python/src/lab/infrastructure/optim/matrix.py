"""Matrix rules: Muon's orthogonalized momentum.

A port of `optimizer_benchmark.matrix.MuonOptimizer`, with its arithmetic.
The hidden matrices are the model's: the weights of its blocks, which for
GPTMini are the matrices the benchmark found by name.
"""

import math

import torch

from .direction import DirectionOptimizer


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
