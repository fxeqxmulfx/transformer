"""Eager column updates, using the lab's data, selection and resume machinery.

Source: matchingMixture_criterion_convex at 5d91bc4. Supporting output
gradients price genuine new heads; no raw model-gradient update is called.
Inner Q/K search is explicitly nonconvex when SearchPricing is selected.
"""

from ..benchmarks.samplers import gather
from ..optim.atomic import AtomicColumns
from .loop import evaluate


class AtomicStepper:
    def __init__(self, experiment, task, model, clock):
        self.task, self.model = task, model
        self.batch = experiment.evaluate.batch
        self.optimizer = AtomicColumns(experiment.optimizer, model, task, experiment.seeds.model)
        self.norms = []

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state):
        self.optimizer.load_state_dict(state)

    def prepare(self, sizes):
        """Eager physical-head shapes require no compilation or warmup."""

    def step(self, parts, rate, sampled):
        batch = self.task.inputs(gather(parts))
        self.model.train()
        loss = self.optimizer.step(batch)
        self.norms.append(None)
        measurements = {"batch_loss": loss, "atomic_columns": self.optimizer.report()} if sampled else None
        return len(batch), measurements

    def gradient_norms(self):
        norms, self.norms = self.norms, []
        return norms

    def evaluate(self, split):
        return evaluate(self.task, self.model, self.task.splits[split], self.batch)
