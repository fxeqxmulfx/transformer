"""Shared update mechanics, including the exact formalized global guard."""

import torch


def guard_accepts(gradients, directions, sigma=0.5):
    """Optimization.Basic.descentGuard, on the joint parameter vector."""
    norm_g = sum(g.square().sum() for g in gradients)
    norm_d = sum(d.square().sum() for d in directions)
    alignment = sum((g * d).sum() for g, d in zip(gradients, directions))
    return bool((alignment >= sigma * norm_g) & (norm_d <= norm_g))


class DirectionOptimizer(torch.optim.Optimizer):
    """Build all directions before updating unique parameter tensors."""

    def __init__(self, named_parameters, lr, *, guarded=False, sigma=0.5):
        named_parameters = list(named_parameters)
        if len({id(p) for _, p in named_parameters}) != len(named_parameters):
            raise ValueError("Duplicate parameters would update tied weights twice")
        super().__init__([p for _, p in named_parameters], {"lr": lr})
        self.names = {p: name for name, p in named_parameters}
        self.guarded = guarded
        self.sigma = sigma
        self.steps = 0
        self.guard_accepted = 0

    def propose(self, parameters):
        raise NotImplementedError

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError("The benchmark evaluates one current minibatch explicitly")
        parameters = [p for p in self.param_groups[0]["params"] if p.grad is not None]
        self.steps += 1
        directions = self.propose(parameters)
        if len(parameters) != len(directions):
            raise ValueError("Missing optimizer directions")
        gradients = [p.grad for p in parameters]
        if self.guarded:
            accepted = guard_accepts(gradients, directions, self.sigma)
            self.guard_accepted += int(accepted)
            if not accepted:
                directions = gradients
        rate = self.param_groups[0]["lr"]
        for parameter, direction in zip(parameters, directions):
            parameter.add_(direction, alpha=-rate)

    def close(self):
        """Release hooks, if the optimizer has installed any."""


def moment_buffers(state, gradient):
    if not state:
        state.update(m=torch.zeros_like(gradient), v=torch.zeros_like(gradient),
                     maximum=torch.zeros_like(gradient))
    return state["m"], state["v"], state["maximum"]


def adam_direction(state, gradient, step, *, beta=0.9, beta2=0.999,
                   eps=1e-8, bias_correction=False):
    m, v, _ = moment_buffers(state, gradient)
    m.mul_(beta).add_(gradient, alpha=1 - beta)
    v.mul_(beta2).addcmul_(gradient, gradient, value=1 - beta2)
    if bias_correction:
        return (m / (1 - beta ** step)) / ((v / (1 - beta2 ** step)).sqrt() + eps)
    return m / (v.sqrt() + eps)
