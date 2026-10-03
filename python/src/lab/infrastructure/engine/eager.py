"""Updates issued kernel by kernel with the native optimizer, as the historical trainer issued them."""

import torch

from ..benchmarks.samplers import gather
from ..optim import build_optimizer
from . import measure
from .loop import evaluate


class EagerStepper:
    def __init__(self, experiment, task, model, clock):
        self.task, self.model, self.clock = task, model, clock
        self.batch = experiment.evaluate.batch
        self.optimizer = build_optimizer(experiment.optimizer, model)
        self.norms = []

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state):
        self.optimizer.load_state_dict(state)

    def prepare(self, sizes):
        """Nothing to prepare: every update is issued as it comes."""

    def step(self, parts, rate, sampled):
        batch = self.task.inputs(gather(parts).to(self.task.device))
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        for group in self.optimizer.param_groups:
            group["lr"] = rate
        output, targets = self.task.forward(self.model, batch)
        self.task.loss(output, targets).backward()
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), float("inf"), error_if_nonfinite=True)
        if sampled:
            with self.clock.diagnosing():
                before = measure.before_update(self.model, self.task, output, targets)
        self.optimizer.step()
        self.norms.append(float(norm))
        if not sampled:
            return len(batch), None
        with self.clock.diagnosing():
            return len(batch), measure.after_update(self.model, self.optimizer, self.task, *before, self.norms[-1])

    def gradient_norms(self):
        norms, self.norms = self.norms, []
        return norms

    def evaluate(self, split):
        return evaluate(self.task, self.model, self.task.splits[split], self.batch)
