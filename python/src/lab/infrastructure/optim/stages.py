"""Stages between a rule's directions and the update.

A stage is called with the optimizer, the parameters and their directions,
and returns the directions to apply. Its state is created with it, in the
optimizer's `state` under the stage's name.
"""

import torch


class Guard:
    """`Transformer.Optimization.descentGuard` on the joint parameter vector.

    The arithmetic of `full_compile_benchmark.step.FullStep`: squared norms
    and the alignment summed over parameters in order, and the decision as a
    device tensor, so a captured update replays it.
    """
    name = "guard"

    def __init__(self, sigma, optimizer):
        self.sigma = sigma
        device = optimizer.param_groups[0]["params"][0].device
        optimizer.state[self.name] = {key: torch.zeros((), dtype=torch.int64, device=device)
                                      for key in ("updates", "accepted")}

    def __call__(self, optimizer, parameters, directions):
        state, gradients = optimizer.state[self.name], [parameter.grad for parameter in parameters]
        norm_g = sum(gradient.square().sum() for gradient in gradients)
        norm_d = sum(direction.square().sum() for direction in directions)
        alignment = sum((gradient * direction).sum() for gradient, direction in zip(gradients, directions))
        accepted = (alignment >= self.sigma * norm_g) & (norm_d <= norm_g)
        state["updates"].add_(1)
        state["accepted"].add_(accepted)
        return [torch.where(accepted, direction, gradient) for gradient, direction in zip(gradients, directions)]

    @staticmethod
    def report(state):
        updates, accepted = state["updates"].item(), state["accepted"].item()
        return {"updates": updates, "accepted": accepted, "acceptance": accepted / max(updates, 1)}
