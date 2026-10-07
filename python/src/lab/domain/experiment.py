"""An experiment: a model, a benchmark and the conditions of one training run.

An experiment file defines `experiments`, a mapping from labels to
`Experiment` values. Variants are derived from a base with `swap`,
`substitute` or `grid`; nothing is configured from the command line.
"""

from collections.abc import Mapping
from dataclasses import dataclass
import re

from .benchmarks import Benchmark, ModularDivision
from .model import Model
from .atomic import (AtomicColumns, AtomicMatching, BindingPricing, MatchingBindings,
                     MatchingOrders, OrderPricing, PairedMatching)
from .optimizers import ANSR, EVD, AdamW, Clipped, Optimizer
from .spec import Spec, describe, require, require_kind, swap, walk
from .stopping import Solved, Stopping
from .synthetic import Synthetic
from .tensor import TensorStack, check_tensor_benchmark
from .training import AttentionDiagnostics, Budget, Checkpoint, CudaGraph, Diagnostics, Eager, Evaluate, Execution, Schedule, Seeds
from .training import FlopBudget, Measured

LABEL = re.compile(r"[a-z0-9][a-z0-9._-]*")


@dataclass(frozen=True)
class Experiment(Spec):
    """Everything that determines a run; its description identifies the run."""
    model: Model
    benchmark: Benchmark
    optimizer: Optimizer
    schedule: Schedule
    budget: Budget
    seeds: Seeds
    evaluate: Evaluate
    execution: Execution
    diagnostics: Diagnostics = Diagnostics()
    checkpoint: Checkpoint = Checkpoint()
    stopping: Stopping | None = None

    def check(self):
        for name, kind in (("model", Model), ("benchmark", Benchmark), ("optimizer", Optimizer),
                           ("schedule", Schedule), ("budget", Budget), ("seeds", Seeds),
                           ("evaluate", Evaluate), ("execution", Execution),
                           ("diagnostics", Diagnostics), ("checkpoint", Checkpoint)):
            require_kind(getattr(self, name), kind, name)
        require(self.benchmark.context <= self.model.context,
                f"The benchmark reads {self.benchmark.context} tokens; "
                f"the model context is {self.model.context}")
        anneal = self.schedule.anneal
        require(anneal is None or anneal.end <= self.budget.updates, "Annealing must end within the budget")
        require(not (isinstance(self.execution, CudaGraph)
                     and any(isinstance(block, EVD) for _, block in walk(self.optimizer))),
                "torch cannot capture an eigendecomposition: run Dash with EVD under Eager()")
        require(not any(isinstance(block, Clipped) for path, block in walk(self.optimizer) if path),
                "Clipping acts on the gradient before the whole rule: write Clipped outermost")
        if isinstance(self.execution, Measured):
            require(isinstance(self.optimizer, AdamW), "Measured execution covers ordinary AdamW")
            require(isinstance(self.benchmark, Synthetic), "Measured execution currently covers synthetic benchmarks")
            require(not isinstance(self.diagnostics, AttentionDiagnostics) and not self.diagnostics.every
                    and not self.diagnostics.neighbors, "Measured runs keep extra arithmetic diagnostics disabled")
        if isinstance(self.budget, FlopBudget):
            require(isinstance(self.execution, Measured), "FlopBudget needs Measured execution")
        population = any(isinstance(block, ANSR) for _, block in walk(self.optimizer))
        if population:
            require(isinstance(self.optimizer, ANSR), "ANSR does not take gradient optimizer stages")
            require(isinstance(self.execution, Eager), "ANSR population evaluation requires Eager execution")
            require(self.schedule == Schedule(), "ANSR has no learning-rate schedule")
            require(not self.diagnostics.gradients, "ANSR computes no gradient norms")
        columns = any(isinstance(block, AtomicColumns) for _, block in walk(self.optimizer))
        if columns:
            require(isinstance(self.optimizer, AtomicColumns), "AtomicColumns must be the whole optimizer")
            require(isinstance(self.model, AtomicMatching), "AtomicColumns needs AtomicMatching")
            require(isinstance(self.execution, Eager), "AtomicColumns requires Eager execution")
            require(self.schedule == Schedule(), "AtomicColumns has no outer learning-rate schedule")
            require(not self.diagnostics.gradients, "AtomicColumns does not update raw model gradients")
            if isinstance(self.optimizer.pricing, OrderPricing):
                require(type(self.benchmark) is MatchingOrders, "OrderPricing needs MatchingOrders")
                require(self.model.width == 1 and self.model.cap == 1 and self.model.channels == 1,
                        "OrderPricing requires scalar heads with cap one")
            if isinstance(self.optimizer.pricing, BindingPricing):
                require(isinstance(self.benchmark, MatchingBindings), "BindingPricing needs MatchingBindings")
                require(isinstance(self.model, PairedMatching) and self.model.cap == 1 and self.model.channels == 1,
                        "BindingPricing requires scalar paired heads with cap one")
            if isinstance(self.optimizer.pricing, (OrderPricing, BindingPricing)):
                require(self.budget.batch == 2, "Exact scalar pricing needs both observations in every update")
                require(self.model.heads <= 3, "The exact scalar QP uses at most three heads")
                require(self.execution.device == "cpu", "The exact scalar QP runs on the CPU")
        if isinstance(self.benchmark, MatchingOrders):
            require(isinstance(self.model, AtomicMatching) and self.model.channels == 1,
                    "MatchingOrders requires scalar AtomicMatching outputs")
        if isinstance(self.model, AtomicMatching):
            require(isinstance(self.execution, Eager), "AtomicMatching currently requires Eager execution")
            require(not isinstance(self.diagnostics, AttentionDiagnostics),
                    "AtomicMatching reports physical heads through the benchmark, not Transformer diagnostics")
            require(columns or isinstance(self.optimizer, AdamW),
                    "AtomicMatching supports AtomicColumns or raw AdamW control")
        if isinstance(self.model, TensorStack):
            check_tensor_benchmark(self.benchmark)
            require(isinstance(self.optimizer, AdamW), "TensorStack uses ordinary AdamW")
            require(not isinstance(self.diagnostics, AttentionDiagnostics),
                    "TensorStack has joint distributions, not the original attention diagnostics")
        if isinstance(self.diagnostics, AttentionDiagnostics):
            require(isinstance(self.benchmark, (Synthetic, ModularDivision)),
                    "AttentionDiagnostics needs a Synthetic or ModularDivision benchmark")
        if self.stopping is not None:
            require_kind(self.stopping, Stopping, "stopping")
            require(self.benchmark.selection is not None,
                    f"{type(self.benchmark).__name__} has no selection split to stop on")
            require(not isinstance(self.stopping, Solved) or self.benchmark.targeted,
                    f"{type(self.benchmark).__name__} sets no target to stop at")


def study(experiments):
    """Validate the `experiments` mapping of an experiment file."""
    if not isinstance(experiments, Mapping) or not experiments:
        raise TypeError("An experiment file defines `experiments`, a nonempty mapping of labels")
    for label, experiment in experiments.items():
        if not isinstance(label, str) or not LABEL.fullmatch(label):
            raise ValueError(f"Label {label!r} must match {LABEL.pattern}")
        require_kind(experiment, Experiment, f"experiments[{label!r}]")
    return dict(experiments)


def grid(base, axes):
    """Every combination of the alternatives on each axis.

    `base` is one experiment, or labeled experiments that each take every
    combination. `axes` maps a dotted path to {label: value}. A variant's
    label joins its base's label and its alternatives' labels with '-', in
    axis order.
    """
    variants = dict(base) if isinstance(base, Mapping) else {"": base}
    for path, options in axes.items():
        require(isinstance(options, Mapping) and options, f"Axis {path} needs labeled alternatives")
        variants = {f"{label}-{option}" if label else option: swap(experiment, path, value)
                    for label, experiment in variants.items() for option, value in options.items()}
    return variants


def differences(old, new, path=""):
    """Dotted paths where two descriptions differ."""
    if isinstance(old, dict) and isinstance(new, dict):
        if old.get("type") != new.get("type"):
            return [path or "<root>"]
        return [change for key in sorted(old.keys() | new.keys())
                for change in differences(old.get(key), new.get(key), f"{path}.{key}" if path else key)]
    if isinstance(old, list) and isinstance(new, list) and len(old) == len(new):
        return [change for index, (a, b) in enumerate(zip(old, new))
                for change in differences(a, b, f"{path}.{index}")]
    return [] if old == new else [path or "<root>"]


def require_continuation(stored, experiment):
    """A run continues only the same experiment, possibly with a larger budget."""
    current = describe(experiment)
    changed = [path for path in differences(stored, current) if path != "budget.updates"]
    if changed:
        raise ValueError("The run directory holds a different experiment; changed: " + ", ".join(changed))
    if experiment.budget.updates < stored["budget"]["updates"]:
        raise ValueError("A continued run may extend its update budget, never shorten it")
