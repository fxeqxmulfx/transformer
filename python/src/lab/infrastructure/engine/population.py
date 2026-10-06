"""Fresh-batch ANSR updates for ordinary lab models.

Source: fxeqxmulfx/ansr examples/benchmark.py at
9cb98c12b3184368c80fea72f3b81432123d96dd. Unlike its fixed copy data, lab
training batches change. The optimizer refreshes personal attractor fitness
before each comparison. There is no backward pass or gradient norm; existing
validation, target stopping, model selection and checkpointing are retained.
"""

from ..benchmarks.samplers import gather
from ..optim import build_optimizer
from .loop import evaluate


class PopulationStepper:
    def __init__(self, experiment, task, model, clock):
        self.task, self.model, self.clock = task, model, clock
        self.batch = experiment.evaluate.batch
        self.optimizer = build_optimizer(experiment.optimizer, model, seed=experiment.seeds.model)
        self.norms = []

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state):
        self.optimizer.load_state_dict(state)

    def prepare(self, sizes):
        """Population shapes do not require compilation or warmup."""

    def step(self, parts, rate, sampled):
        batch = self.task.inputs(gather(parts))
        self.model.train()

        def loss_fn(candidate):
            output, targets = self.task.forward(candidate, batch)
            return self.task.loss(output, targets)

        loss = self.optimizer.step(loss_fn)
        self.norms.append(None)
        measurements = {"batch_loss": float(loss), "population": self.optimizer.report()} if sampled else None
        return len(batch), measurements

    def gradient_norms(self):
        norms, self.norms = self.norms, []
        return norms

    def evaluate(self, split):
        return evaluate(self.task, self.model, self.task.splits[split], self.batch)
