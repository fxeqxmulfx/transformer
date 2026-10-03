"""Magma Algorithm 1 from local arXiv:2602.15322v1, Sections 2–4.

The base computes dense states before independent block masks. Its complete
displacement (including any decay) is multiplied by s*m, without 1/p.
Zero-vector cosine=0 and initial scalar EMA=0.5 are documented extensions.
"""

import math
import torch

from experiments.optimizer_benchmark.common import DirectionOptimizer


def cosine(momentum, gradient):
    numerator = (momentum * gradient).sum()
    denominator = momentum.norm() * gradient.norm()
    safe = torch.where(denominator > 0, denominator, torch.ones_like(denominator))
    return (numerator / safe).clamp(-1, 1)


def eligible(name, parameter):
    return parameter.ndim == 2 and name.startswith("blocks.") and (
        ".attn." in name or ".ffn." in name)


class RMSPropOptimizer(DirectionOptimizer):
    """Section 2 raw v rule; beta2=.999, eps=1e-8, no decay/debiasing.

    Dense beta1=.9 momentum is retained for the wrapper's score only.
    It is not used in this optimizer's parameter direction.
    """

    def __init__(self, named_parameters, lr):
        super().__init__(named_parameters, lr)

    def propose(self, parameters):
        directions = []
        for p in parameters:
            state = self.state[p]
            if not state:
                state.update(m=torch.zeros_like(p), v=torch.zeros_like(p))
            state["m"].mul_(0.9).add_(p.grad, alpha=0.1)
            state["v"].mul_(0.999).addcmul_(p.grad, p.grad, value=0.001)
            directions.append(p.grad / (state["v"].sqrt() + 1e-8))
        return directions


class MagmaOptimizer(DirectionOptimizer):
    """Mask each attention/FFN matrix; keep embedding/temperature updates dense.

    Muon reuses its positive-rescaling-equivalent raw first momentum, not
    its Nesterov direction. SGD needs an extra beta1=.9 scoring EMA.
    """

    def __init__(self, base, *, seed=0, tau=2.0):
        if base.guarded or not math.isfinite(tau) or tau <= 0:
            raise ValueError("Magma requires an unguarded base and finite positive temperature")
        named = [(name, p) for p, name in base.names.items()]
        super().__init__(named, base.param_groups[0]["lr"])
        self.base, self.tau = base, tau
        self.param_groups = base.param_groups
        self.steps = base.steps
        self.masked_parameters = {p for name, p in named if eligible(name, p)}
        self.mask_generator = torch.Generator(device="cpu").manual_seed(20000 + seed)

    def draw_masks(self, count):
        return (torch.rand(count, generator=self.mask_generator) < 0.5).tolist()

    def first_moment(self, p, state):
        dense = self.base.state[p]
        if "momentum" in dense:
            return dense["momentum"]
        if getattr(self.base, "rule", None) != "sgd" and "m" in dense:
            return dense["m"]
        if "alignment_momentum" not in state:
            state["alignment_momentum"] = torch.zeros_like(p)
        return state["alignment_momentum"].mul_(0.9).add_(p.grad, alpha=0.1)

    def propose(self, parameters):
        self.base.steps = self.steps
        directions = self.base.propose(parameters)
        masks = iter(self.draw_masks(sum(p in self.masked_parameters for p in parameters)))
        result = []
        for p, direction in zip(parameters, directions):
            if p not in self.masked_parameters:
                result.append(direction)
                continue
            state = self.state[p]
            if "scale" not in state:
                state.update(scale=p.new_tensor(0.5), scale_sum=p.new_zeros(()),
                             cosine_sum=p.new_zeros(()), draws=0, kept=0)
            alignment = cosine(self.first_moment(p, state), p.grad)
            state["scale"].mul_(0.9).add_(torch.sigmoid(alignment / self.tau), alpha=0.1)
            keep = next(masks)
            state["draws"] += 1
            state["kept"] += int(keep)
            state["scale_sum"].add_(state["scale"])
            state["cosine_sum"].add_(alignment)
            result.append(direction * state["scale"] if keep else torch.zeros_like(direction))
        return result

    def diagnostics(self):
        blocks = {}
        for p in self.param_groups[0]["params"]:
            if p not in self.masked_parameters or "scale" not in self.state[p]:
                continue
            state = self.state[p]
            n = state["draws"]
            blocks[self.names[p]] = {
                "draws": n, "kept": state["kept"], "survival_fraction": state["kept"] / n,
                "mean_damping": state["scale_sum"].item() / n,
                "mean_cosine": state["cosine_sum"].item() / n,
                "final_damping": state["scale"].item()}
        draws = sum(b["draws"] for b in blocks.values())
        return {"blocks": blocks, "mask_draws": draws,
                "survival_fraction": sum(b["kept"] for b in blocks.values()) / draws if draws else None}

    def state_dict(self):
        result = super().state_dict()
        result["magma"] = {"base": self.base.state_dict(), "steps": self.steps,
                           "mask_rng": self.mask_generator.get_state(), "tau": self.tau}
        return result

    def load_state_dict(self, state_dict):
        core = dict(state_dict)
        wrapper = core.pop("magma")
        if wrapper["tau"] != self.tau:
            raise ValueError("Checkpoint uses a different Magma temperature")
        super().load_state_dict(core)
        self.base.load_state_dict(wrapper["base"])
        self.param_groups = self.base.param_groups
        self.steps = self.base.steps = wrapper["steps"]
        self.mask_generator.set_state(wrapper["mask_rng"].cpu())

    def close(self):
        self.base.close()
