"""Compiled updates with audited per-shape reference arithmetic.

Source: plan.md stage 3, CompiledStepper and arithmetic.VERSION. Probes
use independent eager weights and never update the learned model. Actual
compiled updates keep the original AdamW implementation. All literal
candidate layers and zero FFNs are included in the traced arithmetic.
Evaluation counts every actual model call, including both parity calls;
metric aggregation, compilation and profiling probes are reported as
unmetered setup/observation work, rather than assigned to training FLOPs.
"""

import copy
import math

import torch

from ...domain.training import FlopBudget
from ..arithmetic import Arithmetic, VERSION
from ..optim import build_optimizer
from .compiled import CompiledStepper
from .loop import evaluate


class MeteredInference:
    """Count the real call shape, then return the ordinary compiled result."""
    def __init__(self, owner):
        self.owner = owner

    def __getattr__(self, name):
        return getattr(self.owner.inference, name)

    @torch.no_grad()
    def __call__(self, *args, **kwargs):
        signature = tuple((tuple(value.shape), str(value.dtype), str(value.device))
                          for value in (*args, *kwargs.values()) if isinstance(value, torch.Tensor))
        key = repr(signature)
        if key not in self.owner.inference_profiles:
            counter = Arithmetic()
            with counter:
                self.owner.model(*args, **kwargs)
            self.owner.inference_profiles[key] = counter.report()
        profile = self.owner.inference_profiles[key]
        self.owner.evaluation_floating += profile["floating_ops"]
        self.owner.evaluation_integer += profile["integer_ops"]
        return self.owner.inference(*args, **kwargs)


class MeasuredStepper(CompiledStepper):
    def __init__(self, experiment, task, model, clock):
        super().__init__(experiment, task, model, clock)
        self.experiment = experiment
        self.training_profiles, self.inference_profiles = {}, {}
        self.training_floating = self.training_integer = 0
        self.evaluation_floating = self.evaluation_integer = 0
        self.labels = getattr(task, "label_arithmetic", {"floating_ops": 0, "integer_ops": 0, "charged_ops": 0})
        self.charged_labels = self.labels["charged_ops"]
        if isinstance(experiment.budget, FlopBudget) and self.charged_labels > experiment.budget.flops:
            raise ValueError("Complete-label preparation exceeds the declared arithmetic ceiling")
        self.setup_evaluation = 0
        self.metered_inference = MeteredInference(self)
        self.compiled_update, self.update = self.update, self.track_update
        self.last_loss = None

    def track_update(self, batch):
        result = self.compiled_update(batch)
        self.last_loss = result[0].detach()
        return result

    def state_dict(self):
        return {"optimizer": super().state_dict(), "arithmetic": {
            "convention": VERSION, "training_floating": self.training_floating,
            "training_integer": self.training_integer, "evaluation_floating": self.evaluation_floating,
            "evaluation_integer": self.evaluation_integer, "charged_labels": self.charged_labels,
            "setup_evaluation": self.setup_evaluation, "last_loss": self.last_loss}}

    def load_state_dict(self, state):
        super().load_state_dict(state["optimizer"])
        saved = state["arithmetic"]
        if saved["convention"] != VERSION:
            raise ValueError("A measured checkpoint uses another arithmetic convention")
        for name in ("training_floating", "training_integer", "evaluation_floating", "evaluation_integer"):
            setattr(self, name, saved[name])
        self.charged_labels += saved["charged_labels"]
        self.setup_evaluation = saved["setup_evaluation"]
        self.last_loss = saved["last_loss"]
        if isinstance(self.experiment.budget, FlopBudget) and self.training_charged > self.experiment.budget.flops:
            raise ValueError("Restored updates and rebuilt labels exceed the declared arithmetic ceiling")

    def profile_update(self, size):
        """Run exact ordinary update operations on an independent eager copy."""
        model = copy.deepcopy(self.model)
        optimizer = build_optimizer(self.experiment.optimizer, model, None, self.experiment.budget.updates,
                                    self.experiment.seeds.model, fused=True)
        for group in optimizer.param_groups:
            group["lr"] = self.experiment.optimizer.lr
        phases = {}
        counter = Arithmetic()
        with counter:
            batch = self.task.inputs(torch.arange(size), static=True)
            output, targets = self.task.forward(model, batch, supervised=True)
            loss = self.task.loss(output, targets)
        phases["forward_loss"] = counter.report()
        counter = Arithmetic()
        with counter:
            loss.backward()
        phases["backward"] = counter.report()
        counter = Arithmetic()
        with counter:
            parameters = [p for p in model.parameters() if p.requires_grad]
            norm = torch.nn.utils.get_total_norm([p.grad for p in parameters if p.grad is not None])
            if math.isfinite(self.clip):
                torch.nn.utils.clip_grads_with_norm_(parameters, self.clip, norm)
        phases["gradient_norm"] = counter.report()
        counter = Arithmetic()
        with counter:
            optimizer.step()
        phases["adamw"] = counter.report()
        return {"batch": size, "input_length": batch.shape[-1], "phases": phases,
                **{key: sum(phase[key] for phase in phases.values())
                   for key in ("floating_ops", "integer_ops", "charged_ops")}}

    def prepare(self, sizes):
        for size in sizes:
            self.training_profiles[size] = self.profile_update(size)
        floating, integer, loss = self.evaluation_floating, self.evaluation_integer, self.last_loss
        super().prepare(sizes)
        self.setup_evaluation += self.evaluation_floating - floating + self.evaluation_integer - integer
        self.evaluation_floating, self.evaluation_integer = floating, integer
        self.last_loss = loss

    def affords(self, parts):
        if not isinstance(self.experiment.budget, FlopBudget):
            return True
        size = sum(count for _, _, count in parts)
        return self.training_charged + self.training_profiles[size]["charged_ops"] <= self.experiment.budget.flops

    @property
    def training_charged(self):
        return self.training_floating + self.training_integer + self.charged_labels

    def step(self, parts, rate, sampled):
        size, measurements = super().step(parts, rate, sampled)
        profile = self.training_profiles[size]
        self.training_floating += profile["floating_ops"]
        self.training_integer += profile["integer_ops"]
        return size, measurements

    def evaluate(self, split):
        return evaluate(self.task, self.metered_inference, self.task.splits[split], self.batch, static=True)

    def arithmetic(self, detailed=False):
        ceiling = self.experiment.budget.flops if isinstance(self.experiment.budget, FlopBudget) else None
        result = {"convention": VERSION, "training_floating_ops": self.training_floating,
                  "training_integer_ops": self.training_integer, "auxiliary_labels": self.charged_labels,
                  "training_charged_ops": self.training_charged,
                  "evaluation_forward_floating_ops": self.evaluation_floating,
                  "evaluation_forward_integer_ops": self.evaluation_integer,
                  "setup_evaluation_forward_ops": self.setup_evaluation,
                  "last_minibatch_loss": (None if not self.training_floating or self.last_loss is None
                                           else float(self.last_loss)),
                  "ceiling": ceiling, "unspent": None if ceiling is None else ceiling - self.training_charged,
                  "metric_aggregation_and_profiling_metered": False}
        if detailed:
            result.update(training_profiles=self.training_profiles, inference_profiles=self.inference_profiles,
                          label_profile=self.labels)
        return result
