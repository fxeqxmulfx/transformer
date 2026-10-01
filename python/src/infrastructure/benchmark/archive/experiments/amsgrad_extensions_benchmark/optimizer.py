"""Actual AMSGradW and fused AMSGradMD recurrences read from Lean.

MD matches MagnitudeDirection.amsgradMDProposal, including its three
independent AMSGrad memories and old-factor chain rule. The guarded GPT
variant extends checkedProposal to the product of hidden matrices and
ordinary auxiliary tensors; it checks the complete fused displacement.
"""

import math

import torch
from torch.nn import functional as F


def buffers(x):
    return {name: torch.zeros_like(x) for name in ("m", "v", "maximum")}


def amsgrad_step(x, gradient, state, rate, beta, beta2, eps):
    state["m"].mul_(beta).add_(gradient, alpha=1 - beta)
    state["v"].mul_(beta2).addcmul_(gradient, gradient, value=1 - beta2)
    state["maximum"].copy_(torch.maximum(state["maximum"], state["v"]))
    return x - rate * state["m"] / (eps + state["maximum"].sqrt())


def inverse_softplus(gain):
    # log(exp(gain)-1), rewritten to avoid overflow and cancellation.
    return gain + torch.log(-torch.expm1(-gain))


def new_md_state(weight):
    row = weight.new_full((weight.shape[0],), math.log(math.expm1(1.)))
    col = weight.new_full((weight.shape[1],), math.log(math.expm1(1.)))
    radius = torch.linalg.vector_norm(weight.detach()).clone()
    if not bool(torch.isfinite(radius) & (radius > 0)):
        raise ValueError("MD requires a finite nonzero initial matrix")
    return {"raw_row": row, "raw_col": col, "radius": radius,
            "direction": buffers(weight), "row": buffers(row), "col": buffers(col)}


def factor_gradients(direction, raw_row, raw_col, gradient):
    row, col = F.softplus(raw_row), F.softplus(raw_col)
    direction_g = gradient * row[:, None] * col[None, :]
    row_g = (direction * gradient * col[None, :]).sum(1) * raw_row.sigmoid()
    col_g = (direction * gradient * row[:, None]).sum(0) * raw_col.sigmoid()
    return direction_g, row_g, col_g


def md_proposal(weight, state, gradient, direction_rate, gain_rate, beta, beta2, eps):
    row, col = state["raw_row"], state["raw_col"]
    direction = weight / (F.softplus(row)[:, None] * F.softplus(col)[None, :])
    gd, gr, gc = factor_gradients(direction, row, col, gradient)
    candidate = amsgrad_step(direction, gd, state["direction"], direction_rate, beta, beta2, eps)
    next_row = amsgrad_step(row, gr, state["row"], gain_rate, beta, beta2, eps)
    next_col = amsgrad_step(col, gc, state["col"], gain_rate, beta, beta2, eps)
    norm = torch.linalg.vector_norm(candidate)
    # Lean's matrixProject returns zero on a zero candidate. The guarded
    # variant rejects it; no epsilon/clamp alters any nonzero projection.
    scale = torch.where(norm > 0, state["radius"] / torch.where(norm > 0, norm, 1.), 0.)
    fused = candidate * scale * F.softplus(next_row)[:, None] * F.softplus(next_col)[None, :]
    return fused, next_row, next_col


def checked_weights(weights, gradients, candidates, matrix_flags, rate, sigma):
    directions = [(x - y) / rate for x, y in zip(weights, candidates)]
    norm_g = sum(g.square().sum() for g in gradients)
    norm_d = sum(d.square().sum() for d in directions)
    alignment = sum((g * d).sum() for g, d in zip(gradients, directions))
    admissible = (alignment >= sigma * norm_g) & (norm_d <= norm_g)
    halved, chosen = [], []
    for x, g, y, matrix in zip(weights, gradients, candidates, matrix_flags):
        admissible = admissible & y.isfinite().all()
        if matrix:
            admissible = admissible & (y != 0).any()
    for x, g, y, matrix in zip(weights, gradients, candidates, matrix_flags):
        hits_zero = (x - rate * g == 0).all() if matrix else x.new_zeros((), dtype=torch.bool)
        fallback = x - rate * torch.where(hits_zero, .5 * g, g)
        chosen.append(torch.where(admissible, y, fallback))
        halved.append(hits_zero & ~admissible)
    return chosen, admissible, halved


def rebalance(weight, raw_row, raw_col, radius):
    direction = weight / (F.softplus(raw_row)[:, None] * F.softplus(raw_col)[None, :])
    effective_row = torch.linalg.vector_norm(direction) / radius * F.softplus(raw_row)
    return inverse_softplus(effective_row)


class ExtensionOptimizer:
    def __init__(self, parameters, method, rate, *, beta=.9, beta2=.999,
                 eps=1e-8, decay=.01, direction_rate=None, gain_rate=.001,
                 aux_rate=.0003, sigma=.25, counter=None, accepted=None):
        if method not in {"amsgradw", "amsgradmd", "amsgradmd_guarded"}:
            raise ValueError(method)
        if len({id(p) for p in parameters.values()}) != len(parameters):
            raise ValueError("Tied parameters must appear once")
        if rate <= 0 or eps <= 0 or not 0 <= beta < 1 or not 0 <= beta2 <= 1:
            raise ValueError("Invalid AMSGrad coefficients")
        self.parameters, self.method = parameters, method
        self.beta, self.beta2, self.eps, self.decay = beta, beta2, eps, decay
        self.rate, self.gain_rate, self.aux_rate, self.sigma = rate, gain_rate, aux_rate, sigma
        self.guarded = method.endswith("guarded")
        self.direction_rate = direction_rate if direction_rate is not None else (.0003 if self.guarded else rate)
        first = next(iter(parameters.values()))
        self.counter = first.new_zeros((), dtype=torch.float64) if counter is None else counter
        self.accepted = first.new_zeros((), dtype=torch.int64) if accepted is None else accepted
        self.halved = first.new_zeros((), dtype=torch.int64)
        self.md_names = tuple(name for name, p in parameters.items()
                              if method != "amsgradw" and name.startswith("blocks.") and p.ndim == 2)
        self.aux_names = tuple(name for name in parameters if name not in self.md_names)
        self.state = {name: new_md_state(p) if name in self.md_names else {"base": buffers(p)}
                      for name, p in parameters.items()}
        self.param_groups = [{"params": list(parameters.values()), "lr": rate}]

    @torch.no_grad()
    def step(self, gradients, finite=None):
        if finite is None:
            finite = next(iter(self.parameters.values())).new_ones((), dtype=torch.bool)
        candidates, gains = [], {}
        for name, p in self.parameters.items():
            s, g = self.state[name], gradients[name]
            if name in self.md_names:
                q, row, col = md_proposal(p, s, g, self.direction_rate, self.gain_rate,
                                          self.beta, self.beta2, self.eps)
                gains[name] = (row, col)
            else:
                rate = self.rate if self.method == "amsgradw" else self.aux_rate
                q = amsgrad_step(p, g, s["base"], rate, self.beta, self.beta2, self.eps)
                if self.method == "amsgradw":
                    q = q - (rate * self.decay) * p
            candidates.append(q)
        accepted = finite.new_ones(())
        if self.guarded:
            candidates, accepted, halves = checked_weights(list(self.parameters.values()),
                list(gradients.values()), candidates, [n in self.md_names for n in self.parameters],
                self.rate, self.sigma)
            self.accepted.add_((accepted & finite).long())
            self.halved.add_(sum(h.long() for h in halves))
        for (name, p), candidate in zip(self.parameters.items(), candidates):
            if name in self.md_names:
                s = self.state[name]
                row, col = gains[name]
                if self.guarded:
                    repaired = rebalance(candidate, row, col, s["radius"])
                    row = torch.where(accepted, row, repaired)
                s["raw_row"].copy_(torch.where(finite, row, s["raw_row"]))
                s["raw_col"].copy_(torch.where(finite, col, s["raw_col"]))
            p.copy_(torch.where(finite, candidate, p))
        self.counter.add_(1)

    def diagnostics(self):
        matrices = {}
        for name in self.md_names:
            s, p = self.state[name], self.parameters[name]
            row, col = F.softplus(s["raw_row"]), F.softplus(s["raw_col"])
            norm = (p / (row[:, None] * col[None, :])).norm().item()
            radius = s["radius"].item()
            matrices[name] = {"radius": radius, "recovered_norm": norm,
                "relative_sphere_error": abs(norm - radius) / radius,
                "row_gain_min": row.min().item(), "row_gain_max": row.max().item(),
                "col_gain_min": col.min().item(), "col_gain_max": col.max().item()}
        return {"method": self.method, "direction_rate": self.direction_rate,
                "gain_rate": self.gain_rate, "aux_rate": self.aux_rate,
                "decay": self.decay if self.method == "amsgradw" else 0.,
                "md_names": list(self.md_names), "aux_names": list(self.aux_names),
                "accepted_steps": self.accepted.item() if self.guarded else None,
                "halved_matrix_steps": self.halved.item(), "matrices": matrices}

    def close(self):
        pass
