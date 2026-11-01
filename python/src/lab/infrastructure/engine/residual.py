"""Task145 training with full-state conditioning and stable validation stopping.

The final test never enters this loop, including resumed intermediate runs.
Stopping returns the current state, not a retrospectively selected checkpoint.
"""

import copy
from contextlib import nullcontext

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from .compiled import CompiledStepper
from .eager import EagerStepper
from .loop import Training
from .query_conditioning import TransformerQueryConditioner
from ..optim import training_weights
from ..optim.residual_schedule_free import ResidualScheduleFreeAdamW


class QueryConditioning:
    def __init__(self, experiment, task, model, clock):
        self.attention_backend = getattr(experiment.execution, "attention_backend", "auto")
        super().__init__(experiment, task, model, clock)
        self.optimizer = ResidualScheduleFreeAdamW(model.parameters(), **vars(experiment.optimizer))
        self.optimizer.param_groups[0]["lr"] = 0.
        self.conditioner = (TransformerQueryConditioner(model, self.optimizer,
                            compiled=getattr(self, "max_autotune", False))
                            if experiment.model.mixing == "conditioned_qk" else None)

    def attention_policy(self):
        return sdpa_kernel(SDPBackend.MATH) if self.attention_backend == "math" else nullcontext()

    def evaluate(self, split):
        with self.attention_policy():
            return super().evaluate(split)

    def prepare(self, sizes):
        # Compilation evaluates Schedule-Free averages, whose mode round trips
        # can round. Restore the complete saved frame before real updates.
        saved = copy.deepcopy((self.model.state_dict(), self.optimizer.state_dict()))
        training = self.model.training
        try:
            with self.attention_policy():
                return super().prepare(sizes)
        finally:
            self.model.load_state_dict(saved[0])
            self.optimizer.load_state_dict(saved[1])
            self.model.train(training)

    def step(self, parts, rate, sampled):
        if self.conditioner is not None:
            training_weights(self.optimizer)
            self.optimizer.zero_grad(set_to_none=True)
            self.conditioner()
        with self.attention_policy():
            return super().step(parts, rate, sampled)


class AttentionInference(torch.nn.Module):
    """Keep module eval semantics while recording the backend inside inference."""
    def __init__(self, model, backend):
        super().__init__()
        self.model, self.backend = model, backend

    def forward(self, inputs):
        policy = sdpa_kernel(SDPBackend.MATH) if self.backend == "math" else nullcontext()
        with policy:
            return self.model(inputs)


class ResidualCompiledStepper(QueryConditioning, CompiledStepper):
    """Record SDPA selection inside each graph, including inference cache keys."""

    def __init__(self, experiment, task, model, clock):
        super().__init__(experiment, task, model, clock)
        options = {"mode": "max-autotune", "fullgraph": True} if self.max_autotune else {}
        self.inference = torch.compile(AttentionInference(model, self.attention_backend), dynamic=False, **options)

    def forward(self, batch):
        with self.attention_policy():
            return super().forward(batch)


class ResidualEagerStepper(QueryConditioning, EagerStepper):
    """CPU reference with the same state changes and optimizer."""


class ResidualTraining(Training):
    def stopped(self):
        return self.experiment.benchmark.stable(self.history)

    def finish(self):
        step = self.history[-1]["step"]
        # Ordinary Training leaves the final clock segment open on budget exit.
        # Close only an open segment; stable stopping already paused it.
        if not self.stopped():
            self.clock.pause()
        result = {"updates": self.experiment.budget.updates,
                  "stop": {"step": step, "reason": "stable_validation" if self.stopped() else "budget"},
                  "training_seconds": self.clock.training,
                  "diagnostic_seconds": self.clock.diagnostics,
                  "wall_seconds": self.clock.wall(), "final": self.history[-1],
                  "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(self.device)
                       if self.device.type == "cuda" else None,
                  "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(self.device)
                       if self.device.type == "cuda" else None}
        self.run.finish(result)
        return result
