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

import math

import torch
from torch.nn import functional as F

BUFFERS = ("m", "v", "maximum")


def amsgrad(x, gradient, state, prefix, rate, beta, beta2, eps):
    """`Transformer.MagnitudeDirection.amsgradVectorCallback`: the buffers updated, and x's next value."""
    m, v, maximum = (state[prefix + name] for name in BUFFERS)
    m.mul_(beta).add_(gradient, alpha=1 - beta)
    v.mul_(beta2).addcmul_(gradient, gradient, value=1 - beta2)
    maximum.copy_(torch.maximum(maximum, v))
    return x - rate * m / (eps + maximum.sqrt())


def factored_state(weight):
    """Unit gains, the radius of `weight`, and zero buffers for the direction and both gains."""
    radius = torch.linalg.vector_norm(weight.detach()).clone()
    if not bool(torch.isfinite(radius) & (radius > 0)):
        raise ValueError("AMSGradMD factors finite nonzero matrices")
    gain = math.log(math.expm1(1.0))
    rows, columns = weight.shape
    state = {"row": weight.new_full((rows,), gain), "column": weight.new_full((columns,), gain), "radius": radius}
    for prefix, like in (("", weight), ("row.", state["row"]), ("column.", state["column"])):
        state.update({prefix + name: torch.zeros_like(like) for name in BUFFERS})
    return state


def proposal(weight, state, gradient, direction_rate, gain_rate, beta, beta2, eps):
    """`amsgradMDProposal`: the fused candidate and the next raw gains, from gradients at the old factors."""
    row, column = state["row"], state["column"]
    gains_row, gains_column = F.softplus(row), F.softplus(column)
    direction = weight / (gains_row[:, None] * gains_column[None, :])
    direction_gradient = gradient * gains_row[:, None] * gains_column[None, :]
    row_gradient = (direction * gradient * gains_column[None, :]).sum(1) * row.sigmoid()
    column_gradient = (direction * gradient * gains_row[:, None]).sum(0) * column.sigmoid()
    candidate = amsgrad(direction, direction_gradient, state, "", direction_rate, beta, beta2, eps)
    next_row = amsgrad(row, row_gradient, state, "row.", gain_rate, beta, beta2, eps)
    next_column = amsgrad(column, column_gradient, state, "column.", gain_rate, beta, beta2, eps)
    norm = torch.linalg.vector_norm(candidate)
    # The projection of a zero candidate is zero, as in `matrixProject`; the guard rejects it.
    scale = torch.where(norm > 0, state["radius"] / torch.where(norm > 0, norm, 1.0), 0.0)
    fused = candidate * scale * F.softplus(next_row)[:, None] * F.softplus(next_column)[None, :]
    return fused, next_row, next_column


def inverse_softplus(gain):
    """log(exp(gain) - 1), without overflow or cancellation."""
    return gain + torch.log(-torch.expm1(-gain))


def rebalance(weight, row, column, radius):
    """`rebalanceStorage`: the raw row gains under which the direction of `weight` has the radius."""
    direction = weight / (F.softplus(row)[:, None] * F.softplus(column)[None, :])
    return inverse_softplus(torch.linalg.vector_norm(direction) / radius * F.softplus(row))


class AMSGradMD(torch.optim.Optimizer):
    """The `amsgradmd` rule; given the guard's `sigma`, `amsgradmd_guarded`.

    `rate` is the rate the schedule sets: a float, or the one-element device
    tensor of the capturable form, read when an update runs. The direction
    moves at it unless the spec gives the guarded variant its own direction
    rate; the gains and the other parameters move at their constant rates.
    """
    moments = "raw_AMSGrad_buffers_of_directions_and_auxiliary_parameters_for_AMSGradMD"

    def __init__(self, spec, model, rate=None, sigma=None):
        if sigma is None and spec.direction_rate is not None:
            raise ValueError("Unguarded, AMSGradMD moves its directions at its rate; direction_rate is the guard's")
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        super().__init__(parameters, {"lr": 0.0 if rate is None else rate})
        self.spec, self.sigma = spec, sigma
        self.names = {parameter: name for name, parameter in model.named_parameters()}
        self.hidden = set(model.hidden_matrices())
        for parameter in parameters:
            self.state[parameter].update(factored_state(parameter) if parameter in self.hidden
                                         else {name: torch.zeros_like(parameter) for name in BUFFERS})
        if sigma is not None:
            self.state["guard"] = {key: torch.zeros((), dtype=torch.int64, device=parameters[0].device)
                                   for key in ("updates", "accepted", "halved")}

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError("AMSGradMD updates on the gradients in place")
        spec, rate, parameters = self.spec, self.param_groups[0]["lr"], self.param_groups[0]["params"]
        beta, beta2 = spec.betas
        direction_rate = rate if spec.direction_rate is None else spec.direction_rate
        candidates, gains = [], {}
        for parameter in parameters:
            state, gradient = self.state[parameter], parameter.grad
            if parameter in self.hidden:
                candidate, *gains[parameter] = proposal(parameter, state, gradient, direction_rate, spec.gain_rate,
                                                        beta, beta2, spec.eps)
            else:
                candidate = amsgrad(parameter, gradient, state, "", spec.auxiliary_rate, beta, beta2, spec.eps)
            candidates.append(candidate)
        accepted = None if self.sigma is None else self.check(parameters, candidates, rate)
        for parameter, candidate in zip(parameters, candidates, strict=True):
            if parameter in self.hidden:
                state, (row, column) = self.state[parameter], gains[parameter]
                if accepted is not None:
                    row = torch.where(accepted, row, rebalance(candidate, row, column, state["radius"]))
                state["row"].copy_(row)
                state["column"].copy_(column)
            parameter.copy_(candidate)

    def check(self, parameters, candidates, rate):
        """Replace the candidates by the checked ones in place, and return whether the proposal was accepted."""
        gradients = [parameter.grad for parameter in parameters]
        directions = [(parameter - candidate) / rate for parameter, candidate in zip(parameters, candidates)]
        norm_g = sum(gradient.square().sum() for gradient in gradients)
        norm_d = sum(direction.square().sum() for direction in directions)
        alignment = sum((gradient * direction).sum() for gradient, direction in zip(gradients, directions))
        accepted = (alignment >= self.sigma * norm_g) & (norm_d <= norm_g)
        for parameter, candidate in zip(parameters, candidates):
            accepted = accepted & candidate.isfinite().all()
            if parameter in self.hidden:
                accepted = accepted & (candidate != 0).any()
        halved = []
        for index, (parameter, gradient) in enumerate(zip(parameters, gradients)):
            zeroed = ((parameter - rate * gradient == 0).all() if parameter in self.hidden
                      else parameter.new_zeros((), dtype=torch.bool))
            fallback = parameter - rate * torch.where(zeroed, 0.5 * gradient, gradient)
            candidates[index] = torch.where(accepted, candidates[index], fallback)
            halved.append(zeroed & ~accepted)
        guard = self.state["guard"]
        guard["updates"].add_(1)
        guard["accepted"].add_(accepted.long())
        guard["halved"].add_(sum(zeroed.long() for zeroed in halved))
        return accepted

    def report(self):
        """Per hidden matrix, its radius and the factorization it ends with; the guard's counts."""
        matrices = {}
        for parameter in self.param_groups[0]["params"]:
            if parameter not in self.hidden:
                continue
            state = self.state[parameter]
            row, column = F.softplus(state["row"]), F.softplus(state["column"])
            norm, radius = (parameter / (row[:, None] * column[None, :])).norm().item(), state["radius"].item()
            matrices[self.names[parameter]] = {
                "radius": radius, "recovered_norm": norm, "relative_sphere_error": abs(norm - radius) / radius,
                "row_gain_min": row.min().item(), "row_gain_max": row.max().item(),
                "col_gain_min": column.min().item(), "col_gain_max": column.max().item()}
        report = {"magnitude": matrices}
        if self.sigma is not None:
            updates, accepted, halved = (self.state["guard"][key].item() for key in ("updates", "accepted", "halved"))
            report["guard"] = {"updates": updates, "accepted": accepted, "acceptance": accepted / max(updates, 1),
                               "halved": halved}
        return report
