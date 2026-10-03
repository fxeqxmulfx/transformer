"""One training run: updates, observations, records and checkpoints.

The order of operations is that of `paper_reproduction.grokking.train`, so the
historical configurations reproduce its records bit for bit. A stepper only
decides how updates and evaluations reach the device:

    stepper = Stepper(experiment, task, model, splits, clock)
    stepper.state_dict(), stepper.load_state_dict(state)   the optimizer state
    stepper.prepare()                    once any checkpoint is loaded
    stepper.step(parts, rate, sampled)   one update: (batch size, measurements or None)
    stepper.gradient_norms()             the norms of the updates since the last call
    stepper.evaluate(split)              the metrics of the named split

A stepper may return gradient norms late; the loop collects them before each
observation and stops at the first that is not finite.
"""

from contextlib import contextmanager
import math
import time

import torch

from ...domain import cadence
from ...domain.training import rate
from ..benchmarks import build_task
from ..nn import build_model
from .sampler import EpochSampler


class Clock:
    """Seconds spent training, measuring and in all, accumulated across sessions.

    Training time excludes observations and diagnostics, as the historical
    trainer counted it; the device is synchronized at every boundary.
    """

    def __init__(self, device):
        self.device = device
        self.started = time.perf_counter()
        self.training, self.diagnostics, self.earlier = 0.0, 0.0, 0.0
        self.segment, self.segment_diagnostics = None, 0.0

    def restore(self, checkpoint):
        self.training, self.diagnostics = checkpoint["training_seconds"], checkpoint["diagnostic_seconds"]
        self.earlier = checkpoint["wall_seconds"]

    def synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def wall(self):
        return self.earlier + time.perf_counter() - self.started

    def resume(self):
        self.synchronize()
        self.segment = time.perf_counter()

    def pause(self):
        self.synchronize()
        self.training += time.perf_counter() - self.segment - self.segment_diagnostics
        self.diagnostics += self.segment_diagnostics
        self.segment_diagnostics = 0.0

    @contextmanager
    def diagnosing(self):
        self.synchronize()
        started = time.perf_counter()
        yield
        self.segment_diagnostics += time.perf_counter() - started


@torch.no_grad()
def evaluate(task, model, rows, batch):
    """The metrics of one split, evaluated exhaustively in chunks of `batch` rows."""
    model.eval()
    sums = torch.zeros(task.sums, dtype=torch.float64, device=rows.device)
    for start in range(0, len(rows), batch):
        task.accumulate(model, rows[start:start + batch], sums)
    return task.metrics(sums.tolist(), len(rows))


class Training:
    """Train an experiment from its run's last checkpoint, or from scratch, to the budget."""

    def __init__(self, experiment, run, progress, stepper):
        self.experiment, self.run, self.progress = experiment, run, progress
        self.device = torch.device(experiment.execution.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable")
        torch.set_num_threads(experiment.execution.threads)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        self.clock = Clock(self.device)
        self.task = build_task(experiment.benchmark, experiment.seeds.data)
        self.model = build_model(experiment.model, self.task.vocab, experiment.seeds.model).to(self.device)
        self.splits = {name: torch.tensor(rows, dtype=torch.long, device=self.device)
                       for name, rows in self.task.splits.items()}
        self.stepper = stepper(experiment, self.task, self.model, self.splits, self.clock)
        self.sampler = EpochSampler(len(self.splits["train"]), experiment.budget, experiment.seeds.batch_seed)
        self.completed, self.seen, self.last_batch_size = 0, 0, None
        checkpoint = run.checkpoint(self.device)
        if checkpoint is not None:
            self.model.load_state_dict(checkpoint["model"])
            self.stepper.load_state_dict(checkpoint["optimizer"])
            self.sampler.restore(checkpoint)
            self.clock.restore(checkpoint)
            self.completed, self.seen = checkpoint["step"], checkpoint["examples_seen"]
            self.last_batch_size = checkpoint["last_batch_size"]
        # Records past the checkpoint are dropped; without a checkpoint, every record is.
        run.rewind(self.completed if checkpoint is not None else -1)
        self.history = run.records("history")
        if experiment.diagnostics.gradients and (
                [row["step"] for row in run.records("gradients")] != list(range(1, self.completed + 1))):
            raise ValueError("The gradient trace does not cover every checkpointed update")
        self.trace = []
        self.stepper.prepare()

    def observe(self, step, probe):
        row = {"step": step, "epochs_seen": self.seen / len(self.splits["train"]),
               "training_seconds": self.clock.training, "wall_seconds": self.clock.wall(),
               "last_batch_size": self.last_batch_size, **{name: self.stepper.evaluate(name) for name in self.splits}}
        if probe:
            self.run.record("probes", row)
        else:
            self.history.append(row)
            self.run.record("history", row)
        self.progress({"diagnostic_probe": True, **row} if probe else row)

    def flush(self):
        """Check, and record if traced, the gradient norms of the updates since the last flush."""
        for row, norm in zip(self.trace, self.stepper.gradient_norms(), strict=True):
            if not math.isfinite(norm):
                raise FloatingPointError(f"The gradient norm of update {row['step']} is {norm}")
            if self.experiment.diagnostics.gradients:
                self.run.record("gradients", {**row, "gradient_l2": norm})
        self.trace = []

    def save(self, step):
        self.run.save({"model": self.model.state_dict(), "optimizer": self.stepper.state_dict(), "step": step,
                       "examples_seen": self.seen, "training_seconds": self.clock.training,
                       "diagnostic_seconds": self.clock.diagnostics, "last_batch_size": self.last_batch_size,
                       "wall_seconds": self.clock.wall(), **self.sampler.state()})

    def __call__(self):
        experiment = self.experiment
        if not self.history:
            self.observe(0, probe=False)
        self.clock.resume()
        for step in range(self.completed + 1, experiment.budget.updates + 1):
            parts, place = self.sampler.next()
            learning_rate = rate(experiment.optimizer.lr, experiment.schedule, step - 1)
            sampled = cadence.sampled(experiment, step)
            size, measurements = self.stepper.step(parts, learning_rate, sampled)
            self.seen += size
            self.last_batch_size = size
            self.trace.append({"step": step, "batch_size": size, "epoch_tail": place["epoch_tail"],
                               "learning_rate": learning_rate})
            if sampled:
                with self.clock.diagnosing():
                    self.run.record("diagnostics", {"step": step, "batch_size": size, **place,
                                                    "learning_rate": learning_rate, **measurements})
            canonical = cadence.canonical(experiment, step)
            if canonical or cadence.probe(experiment, step):
                with self.clock.diagnosing():
                    self.flush()
                self.clock.pause()
                self.observe(step, probe=not canonical)
                if cadence.checkpointed(experiment, step):
                    self.save(step)
                self.clock.resume()
        memory = {"peak_cuda_allocated_bytes": None, "peak_cuda_reserved_bytes": None}
        if self.device.type == "cuda":
            memory = {"peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(self.device),
                      "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(self.device)}
        result = {"updates": experiment.budget.updates, "training_seconds": self.clock.training,
                  "diagnostic_seconds": self.clock.diagnostics, "wall_seconds": self.clock.wall(), **memory,
                  **self.task.analyze(self.history), "final": self.history[-1]}
        self.run.finish(result)
        return result
