"""Muon and blocked/batched DASH, with the formalized optional guard."""

from collections import defaultdict
import math
import torch

from .common import DirectionOptimizer, adam_direction
from .roots import fitted_power, inverse_fourth_cn, inverse_fourth_evd, inverse_fourth_ndb


def muon_polar(gradient, steps=5):
    transpose = gradient.size(0) > gradient.size(1)
    x = gradient.T if transpose else gradient
    norm = x.square().sum().sqrt()
    x = x / torch.where(norm > 0, norm, torch.ones_like(norm))
    for _ in range(steps):
        gram = x @ x.T
        x = 3.4445 * x + (-4.775 * gram + 2.0315 * (gram @ gram)) @ x
    return x.T if transpose else x


class MuonOptimizer(DirectionOptimizer):
    def __init__(self, named_parameters, lr, *, guarded=False, momentum=0.95):
        super().__init__(named_parameters, lr, guarded=guarded)
        self.momentum = momentum

    def propose(self, parameters):
        directions = []
        for p in parameters:
            state = self.state[p]
            # Paper convention: AdamW on the tied embedding and scalar parameters.
            if p.ndim != 2 or self.names[p] == "embed.weight":
                direction = 0.05 * adam_direction(state, p.grad, self.steps, bias_correction=True)
            else:
                if not state:
                    state["momentum"] = torch.zeros_like(p)
                moment = state["momentum"]
                moment.mul_(self.momentum).add_(p.grad)
                nesterov = self.momentum * moment + p.grad
                direction = 0.2 * math.sqrt(max(p.shape)) * muon_polar(nesterov)
            directions.append(direction)
        return directions


class DashOptimizer(DirectionOptimizer):
    def __init__(self, named_parameters, lr, *, solver="ndb", guarded=False,
                 block_size=32, beta=0.99, graft_beta=0.9, graft_beta2=0.999,
                 epsilon=1e-4, root_steps=6):
        super().__init__(named_parameters, lr, guarded=guarded)
        if solver not in {"ndb", "cn", "evd", "chebyshev"}:
            raise ValueError(solver)
        self.solver, self.beta, self.epsilon = solver, beta, epsilon
        self.graft_beta, self.graft_beta2, self.root_steps = graft_beta, graft_beta2, root_steps
        plans = defaultdict(list)
        for p in self.param_groups[0]["params"]:
            rows, columns = p.shape if p.ndim == 2 else (p.numel(), 1)
            for i in range(0, rows, block_size):
                for j in range(0, columns, block_size):
                    height, width = min(block_size, rows - i), min(block_size, columns - j)
                    plans[height, width].append((p, i, j))
        self.groups = []
        for (height, width), entries in plans.items():
            p = entries[0][0]
            zeros = lambda *shape: torch.zeros(shape, device=p.device, dtype=p.dtype)
            count = len(entries)
            self.groups.append({"shape": (height, width), "entries": entries,
                                "left": zeros(count, height, height),
                                "right": zeros(count, width, width),
                                "moment": zeros(count, height, width),
                                "accumulator": zeros(count, height, width)})

    def roots(self, history):
        if self.solver == "chebyshev":
            return fitted_power(history, self.epsilon)
        identity = torch.eye(history.size(-1), device=history.device, dtype=history.dtype)
        matrix = history + self.epsilon * identity
        if self.solver == "ndb":
            return inverse_fourth_ndb(matrix, self.root_steps)
        if self.solver == "cn":
            return inverse_fourth_cn(matrix, self.root_steps + 2)
        return inverse_fourth_evd(matrix)

    def propose(self, parameters):
        directions = {p: torch.zeros_like(p) for p in parameters}
        for group in self.groups:
            height, width = group["shape"]
            gradients = []
            for p, i, j in group["entries"]:
                if p.grad is None:
                    raise ValueError("DASH benchmark expects dense gradients for every parameter")
                matrix = p.grad if p.ndim == 2 else p.grad.view(-1, 1)
                gradients.append(matrix[i:i + height, j:j + width])
            gradient = torch.stack(gradients)
            transpose = gradient.transpose(-2, -1)
            group["left"].mul_(self.beta).add_(gradient @ transpose, alpha=1 - self.beta)
            group["right"].mul_(self.beta).add_(transpose @ gradient, alpha=1 - self.beta)
            group["moment"].mul_(self.graft_beta).add_(gradient, alpha=1 - self.graft_beta)
            group["accumulator"].mul_(self.graft_beta2).addcmul_(
                gradient, gradient, value=1 - self.graft_beta2)
            reference = group["moment"] / (group["accumulator"].sqrt() + self.epsilon)
            output = self.roots(group["left"]) @ gradient @ self.roots(group["right"])
            norm = output.square().sum(dim=(-2, -1)).sqrt()
            reference_norm = reference.square().sum(dim=(-2, -1)).sqrt()
            # Lean's total division gives zero at a zero Shampoo direction.
            denominator = torch.where(norm > 0, norm, torch.ones_like(norm))
            output = output * (reference_norm / denominator)[..., None, None]
            for value, (p, i, j) in zip(output.unbind(), group["entries"]):
                destination = directions[p] if p.ndim == 2 else directions[p].view(-1, 1)
                destination[i:i + height, j:j + width] = value
        return [directions[p] for p in parameters]
