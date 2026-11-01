"""Finite sign-vector tasks with analytic targets, disjoint inputs and sealed test.

Source: domain.gradient_paths. Relative MSE divides by teacher energy; gain
error and a worst-case coefficient bound prevent tiny labels hiding failure.
"""

import hashlib

import torch
from torch.nn import functional as F

from ...domain.gradient_paths import BranchGates
from .samplers import EpochSampler


class GradientTask:
    components = ("loss",)

    def __init__(self, spec, data_seed, device, model_spec):
        self.spec, self.model_spec, self.device, self.vocab = spec, model_spec, device, 0
        generator = torch.Generator().manual_seed(data_seed)
        seen, rows = set(), []
        for _ in range(spec.train + spec.validation + spec.test):
            while True:
                signs = torch.randint(2, (model_spec.context, model_spec.width), generator=generator)
                key = bytes(signs.flatten().tolist())
                if key not in seen:
                    seen.add(key)
                    break
            rows.append(signs.float() * 2 - 1)
        inputs = torch.stack(rows)
        self.teacher = (torch.tensor([(.25, .5, .75)[i % 3] for i in range(model_spec.depth)])
                        if isinstance(model_spec, BranchGates) else torch.tensor([.25]))
        targets = (inputs[:, 0] + (inputs[:, 1:] * self.teacher[None, :, None]).sum(1)
                   if isinstance(model_spec, BranchGates) else inputs[:, 0] * .25)
        packed = torch.cat([inputs, targets[:, None]], dim=1).to(device)
        a, b = spec.train, spec.train + spec.validation
        self.splits = {"train": packed[:a], "validation": packed[a:b], "test": packed[b:]}
        self.hashes = {name: hashlib.sha256(value.cpu().numpy().tobytes()).hexdigest()
                       for name, value in self.splits.items()}

    def sampler(self, batch, seed):
        return EpochSampler(len(self.splits["train"]), batch, False, seed)

    def inputs(self, indices, static=False):
        return self.splits["train"].index_select(0, indices.to(self.device))

    def forward(self, model, batch, supervised=False):
        return model(batch[:, :-1]), batch[:, -1]

    @staticmethod
    def loss(output, targets):
        return F.mse_loss(output, targets)

    @staticmethod
    def position_losses(output, targets):
        return F.mse_loss(output, targets).reshape(1)

    def accumulator(self):
        return torch.zeros(2, dtype=torch.float64, device=self.device)

    def accumulate(self, model, chunk, sums, static):
        output, target = self.forward(model, chunk)
        sums.add_(torch.stack([(output.double() - target.double()).square().sum(),
                               target.double().square().sum()]))

    def metrics(self, sums, rows):
        error, energy = sums
        return {"loss": error / (len(rows) * self.model_spec.width),
                "relative_mse": error / energy, "examples": len(rows)}

    def progress(self, seen):
        return {"epochs_seen": seen / self.spec.train}

    def observe(self, model, batch):
        # Training restores y after measuring averaged x. Label that distinction.
        with torch.no_grad():
            coefficient = model.coefficients().cpu()
            errors = (coefficient - self.teacher).abs()
        return {"training_query_coefficients": coefficient.tolist(),
                "training_query_coefficient_error": errors.max().item(),
                "training_query_uniform_l2_bound": self.model_spec.width ** .5 * errors.sum().item()}

    def inspect(self, model, batch):
        return {}

    @staticmethod
    def analyze(history):
        return {"first_validation_relative_mse_1e4": next(
            (row["step"] for row in history if row["validation"]["relative_mse"] <= 1e-4), None)}
