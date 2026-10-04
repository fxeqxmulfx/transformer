"""Updates and evaluations compiled by TorchInductor.

An update reads its batch at the full width of the training split, so each
batch size the sampler draws compiles one forward and loss, and AOTAutograd
the backward of it; the readout reads only the positions a target can read
(`supervised`), and the gradient norm, any clipping and the native AdamW,
fused, run as they come. An evaluation runs the model's forward compiled
for inference, static (`benchmarks`), so the chunks of a split keep their
shapes from one observation to the next; each chunk, and each step of a
generation, compiles its own shape, faster than one compiled for all shapes.
`prepare` compiles the update of every batch size and the evaluation of
every observed split before the clock starts; any other shape compiles when
it first comes, up to `SHAPES` of a function, past which Dynamo runs it
eagerly. A stepper first drops what earlier ones compiled in the process:
their graphs guard on another model, and would only count toward the limit.

A sampled update runs the compiled update as every other update does, so
measurements see what any update leaves. Inductor's kernels compute the
same values on every call at a fixed thread count, so an interrupted run
resumes onto its own records.
"""

import math

import torch

from ...domain.optimizers import clipping
from ..benchmarks.samplers import gather
from ..optim import build_optimizer
from . import measure
from .loop import evaluate

SHAPES = 64


class CompiledStepper:
    def __init__(self, experiment, task, model, clock):
        torch._dynamo.reset()
        torch._dynamo.config.recompile_limit = SHAPES
        self.task, self.model, self.clock = task, model, clock
        self.observed, self.batch = experiment.benchmark.observed, experiment.evaluate.batch
        self.clip = clipping(experiment.optimizer)
        self.optimizer = build_optimizer(experiment.optimizer, model, None, experiment.budget.updates,
                                         experiment.seeds.model, fused=True)
        self.parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        self.update = torch.compile(self.forward, dynamic=False)
        self.inference = torch.compile(model, dynamic=False)
        self.norms = []

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state):
        self.optimizer.load_state_dict(state)

    def forward(self, batch):
        """A batch's loss, its supervised logits and their targets."""
        output, targets = self.task.forward(self.model, batch, supervised=True)
        return self.task.loss(output, targets), output, targets

    def prepare(self, sizes):
        """Compile the update of each batch size and the evaluation of each observed split, changing nothing."""
        self.model.train()
        for size in sizes:
            loss, _, _ = self.update(self.task.inputs(torch.arange(size), static=True))
            loss.backward()
            self.optimizer.zero_grad(set_to_none=True)
        for name in self.observed:
            if self.task.splits[name] is not None:
                self.evaluate(name)

    def step(self, parts, rate, sampled):
        batch = self.task.inputs(gather(parts), static=True)
        if not self.model.training:
            self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        for group in self.optimizer.param_groups:
            group["lr"] = rate
        loss, output, targets = self.update(batch)
        loss.backward()
        norm = torch.nn.utils.get_total_norm([parameter.grad for parameter in self.parameters
                                              if parameter.grad is not None])
        if math.isfinite(self.clip):
            torch.nn.utils.clip_grads_with_norm_(self.parameters, self.clip, norm)
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
        return evaluate(self.task, self.inference, self.task.splits[split], self.batch, static=True)
