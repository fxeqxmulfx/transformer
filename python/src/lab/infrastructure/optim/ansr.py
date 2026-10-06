"""Gradient-free population search, adapted from fxeqxmulfx/ansr.

Source: src/ansr/ansr_torch.py at 9cb98c12b3184368c80fea72f3b81432123d96dd.
Copyright 2025 fxeqxmulfx; MIT license in ansr.LICENSE beside this module.
Mutation, coordinate-wise neighbour choice, clipping, and collision restarts
follow the source. Deviations: refresh attractor fitness on the current
training batch, deterministic gradient-free functional evaluation, persistent
population/RNG checkpoints, and explicit evaluation counters. Initialization
rejects an out-of-box model instead of clipping it. No gradients, learning
rate or weight decay are used. The model's tied parameters remain tied.
"""

import torch
from torch.func import functional_call, vmap


class ANSR(torch.optim.Optimizer):
    def __init__(self, spec, model, seed):
        parameters = list(model.parameters())
        super().__init__(parameters, {})
        self.spec, self.model = spec, model
        self.device = parameters[0].device
        self.names = [name for name, _ in model.named_parameters()]
        self.shapes = [parameter.shape for parameter in parameters]
        self.buffers = dict(model.named_buffers())
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.population = None

    def flat(self):
        return torch.cat([parameter.detach().flatten() for parameter in self.model.parameters()])

    def initialize(self):
        original, spec = self.flat(), self.spec
        if not torch.isfinite(original).all() or original.abs().max() > spec.bound:
            raise ValueError("The initialized model must fit ANSR's finite parameter box")
        positions = torch.rand(spec.popsize, original.numel(), device=self.device, generator=self.generator)
        positions[0] = (original + spec.bound) / (2 * spec.bound)
        self.population = {"pos": positions, "best_pos": positions.clone(),
                           "best_res": torch.full((spec.popsize,), torch.inf, device=self.device),
                           "updates": 0, "nfev": 0, "refresh_nfev": 0, "restarts": 0}

    def state_dict(self):
        return {**super().state_dict(), "population": self.population, "generator": self.generator.get_state()}

    def load_state_dict(self, state):
        super().load_state_dict(state)
        population = state["population"]
        self.population = (None if population is None else
                           {key: value.to(self.device) if isinstance(value, torch.Tensor) else value
                            for key, value in population.items()})
        self.generator.set_state(state["generator"].cpu())

    def evaluate(self, positions, loss_fn):
        """Evaluate candidates without writing the module or constructing gradients."""
        real = positions * (2 * self.spec.bound) - self.spec.bound

        def candidate_loss(flat):
            params, offset = {}, 0
            for name, shape in zip(self.names, self.shapes, strict=True):
                count = shape.numel()
                params[name] = flat[offset:offset + count].view(shape)
                offset += count

            def candidate(*args, **kwargs):
                return functional_call(self.model, (params, self.buffers), args, kwargs)

            return loss_fn(candidate)

        values = [vmap(candidate_loss, randomness="error")(chunk)
                  for chunk in real.split(self.spec.batch_size)]
        self.population["nfev"] += len(positions)
        return torch.cat(values)

    @torch.no_grad()
    def install(self, unit):
        flat, offset = unit * (2 * self.spec.bound) - self.spec.bound, 0
        for parameter in self.model.parameters():
            count = parameter.numel()
            parameter.copy_(flat[offset:offset + count].view_as(parameter))
            offset += count

    @torch.no_grad()
    def step(self, loss_fn):
        """Refresh and compare all candidates on one batch; issue the next generation."""
        if self.population is None:
            self.initialize()
        state, spec = self.population, self.spec
        pos, best_pos, best_res = state["pos"], state["best_pos"], state["best_res"]
        valid = torch.isfinite(best_res)
        if valid.any():
            refreshed = self.evaluate(best_pos[valid], loss_fn)
            best_res[valid] = torch.where(torch.isfinite(refreshed), refreshed, torch.inf)
            state["refresh_nfev"] += int(valid.sum())
        current = self.evaluate(pos, loss_fn)
        improved = torch.isfinite(current) & (current < best_res)
        best_res[improved], best_pos[improved] = current[improved], pos[improved]
        if not torch.isfinite(best_res).any():
            raise FloatingPointError("ANSR found no finite candidate loss on the current batch")
        winner = torch.argmin(best_res)
        result = best_res[winner].clone()

        ii, jj = torch.triu_indices(spec.popsize, spec.popsize, offset=1, device=self.device)
        ri, rj = best_res[ii], best_res[jj]
        high, low = torch.maximum(ri, rj), torch.minimum(ri, rj)
        collided = torch.isfinite(high) & (high != 0) & ((high - low) / high.abs() < spec.restart_tolerance)
        if collided.any():
            i_wins = (ii == winner) | ((jj != winner) & (ri < rj))
            losers = torch.unique(torch.where(i_wins, jj, ii)[collided])
            best_res[losers] = torch.inf
            best_pos[losers] = torch.rand(len(losers), pos.size(1), device=self.device, generator=self.generator)
            pos[losers] = torch.rand(len(losers), pos.size(1), device=self.device, generator=self.generator)
            state["restarts"] += len(losers)

        noise = torch.randn(pos.shape, device=self.device, generator=self.generator) * spec.sigma
        if spec.p_self == 1:
            guide = best_pos
        else:
            use_self = (None if spec.p_self == 0 else
                        torch.rand(pos.shape, device=self.device, generator=self.generator) <= spec.p_self)
            neighbours = torch.randint(0, spec.popsize - 1, pos.shape, device=self.device, generator=self.generator)
            neighbours += (neighbours >= torch.arange(spec.popsize, device=self.device)[:, None]).long()
            guide = best_pos[neighbours, torch.arange(pos.size(1), device=self.device)]
            if use_self is not None:
                guide = torch.where(use_self, best_pos, guide)
        pos.copy_((guide + (guide - pos).abs() * noise).clamp(0, 1))
        self.install(best_pos[winner])
        state["updates"] += 1
        return result

    def report(self):
        if self.population is None:
            return {}
        state = self.population
        return {"algorithm": "ANSR", "parameters": state["pos"].size(1), "popsize": self.spec.popsize,
                "sigma": self.spec.sigma, "p_self": self.spec.p_self, "bound": self.spec.bound,
                **{key: state[key] for key in ("updates", "nfev", "refresh_nfev", "restarts")},
                "restart_frequency": state["restarts"] * self.spec.popsize / state["nfev"],
                "restarts_per_generation": state["restarts"] / state["updates"]}
