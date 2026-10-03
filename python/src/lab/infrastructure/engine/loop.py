"""One training run: updates, observations, records and checkpoints.

The order of operations is that of `paper_reproduction.grokking.train`, so the
historical configurations reproduce its records bit for bit. A stepper only
decides how updates and evaluations reach the device:

    stepper = Stepper(experiment, task, model, clock)
    stepper.state_dict(), stepper.load_state_dict(state)   the optimizer state
    stepper.prepare(sizes)               once any checkpoint is loaded; sizes recur
    stepper.step(parts, rate, sampled)   one update: (batch size, measurements or None)
    stepper.gradient_norms()             the norms of the updates since the last call
    stepper.evaluate(split)              the metrics of the named split
    stepper.optimizer                    whose stages report at the end of the run

A stepper may return gradient norms late; the loop collects them before each
observation and each sampled update, and stops at the first that is not
finite: the run fails, or, under a stopping policy, ends there.

An observation evaluates each observed split once, however many names share
its rows; a split without rows measures None. The task adds what it measures
of the model beyond the splits (`observe`).

When the benchmark has a selection split, the model of the best observation
travels in every checkpoint. At the end of a run the benchmark's last splits
are evaluated once on the last model, which the task inspects too; then the
best model is restored, its selection split is evaluated again and must
measure what was observed, its final splits are evaluated once, and the task
inspects it.
"""

from contextlib import contextmanager
import math
import time

import torch

from ...domain import cadence
from ...domain.stopping import Selection
from ...domain.training import rate
from ..benchmarks import build_task
from ..nn import build_model
from ..optim import report


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


class NonfiniteGradient(FloatingPointError):
    def __init__(self, step, norm):
        super().__init__(f"The gradient norm of update {step} is {norm}")
        self.step = step


@torch.no_grad()
def evaluate(task, model, rows, batch, static=False):
    """The metrics of one split, evaluated exhaustively in chunks of `batch` rows, `static` or not."""
    model.eval()
    sums = task.accumulator()
    for start in range(0, len(rows), batch):
        task.accumulate(model, rows[start:start + batch], sums, static)
    return task.metrics(sums.tolist(), rows)


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
        self.task = build_task(experiment.benchmark, experiment.seeds.data, self.device)
        self.model = build_model(experiment.model, self.task.vocab, experiment.seeds.model).to(self.device)
        self.stepper = stepper(experiment, self.task, self.model, self.clock)
        self.sampler = self.task.sampler(experiment.budget.batch, experiment.seeds.batch_seed)
        self.completed, self.seen, self.last_batch_size = 0, 0, None
        selection = experiment.benchmark.selection
        self.selection = None if selection is None else Selection(experiment.stopping, experiment.benchmark.rank,
                                                                  experiment.benchmark.solved)
        self.best = None
        checkpoint = run.checkpoint(self.device)
        if checkpoint is not None:
            self.model.load_state_dict(checkpoint["model"])
            self.stepper.load_state_dict(checkpoint["optimizer"])
            self.sampler.restore(checkpoint)
            self.clock.restore(checkpoint)
            self.completed, self.seen = checkpoint["step"], checkpoint["examples_seen"]
            self.last_batch_size = checkpoint["last_batch_size"]
            self.best = checkpoint.get("best")
        # Records past the checkpoint are dropped; without a checkpoint, every record is.
        run.rewind(self.completed if checkpoint is not None else -1)
        self.history = run.records("history")
        if self.selection is not None:
            for row in self.history:
                self.selection.observe(row["step"], row[selection])
            if self.selection.step != (None if self.best is None else self.best["step"]):
                raise ValueError("The checkpointed best model is not the best observation of the history")
        if experiment.diagnostics.gradients and (
                [row["step"] for row in run.records("gradients")] != list(range(1, self.completed + 1))):
            raise ValueError("The gradient trace does not cover every checkpointed update")
        self.trace = []
        self.stepper.prepare(self.sampler.sizes())

    def measure(self):
        """The metrics of each observed split, evaluated once for all the names that share its rows."""
        metrics, evaluated = {}, {}
        for name in self.experiment.benchmark.observed:
            rows = self.task.splits[name]
            if rows is not None and id(rows) not in evaluated:
                evaluated[id(rows)] = self.stepper.evaluate(name)
            metrics[name] = None if rows is None else evaluated[id(rows)]
        return metrics

    def observe(self, step, probe):
        row = {"step": step, **self.task.progress(self.seen),
               "training_seconds": self.clock.training, "wall_seconds": self.clock.wall(),
               "last_batch_size": self.last_batch_size, **self.measure(),
               **self.task.observe(self.model, self.experiment.evaluate.batch)}
        if probe:
            self.run.record("probes", row)
        else:
            self.history.append(row)
            self.run.record("history", row)
            selection = self.experiment.benchmark.selection
            if self.selection is not None and self.selection.observe(step, row[selection]):
                self.best = {"step": step, "model": {name: value.detach().cpu().clone()
                                                     for name, value in self.model.state_dict().items()}}
        self.progress({"diagnostic_probe": True, **row} if probe else row)

    def flush(self):
        """Check, and record if traced, the gradient norms of the updates since the last flush."""
        trace, self.trace = self.trace, []
        for row, norm in zip(trace, self.stepper.gradient_norms(), strict=True):
            if not math.isfinite(norm):
                raise NonfiniteGradient(row["step"], norm)
            if self.experiment.diagnostics.gradients:
                self.run.record("gradients", {**row, "gradient_l2": norm})

    def save(self, step):
        self.run.save({"model": self.model.state_dict(), "optimizer": self.stepper.state_dict(), "step": step,
                       "examples_seen": self.seen, "training_seconds": self.clock.training,
                       "diagnostic_seconds": self.clock.diagnostics, "last_batch_size": self.last_batch_size,
                       "wall_seconds": self.clock.wall(), **self.sampler.state(),
                       **({} if self.selection is None else {"best": self.best})})

    def stopped(self):
        return self.selection is not None and self.selection.stop is not None

    def __call__(self):
        if not self.history:
            self.observe(0, probe=False)
        try:
            if not self.stopped():
                self.train()
        except NonfiniteGradient as failure:
            if self.experiment.stopping is None:
                raise
            self.clock.pause()
            self.selection.stop = (failure.step, "nonfinite_gradient")
        return self.finish()

    def train(self):
        experiment = self.experiment
        self.clock.resume()
        for step in range(self.completed + 1, experiment.budget.updates + 1):
            parts, place = self.sampler.next()
            learning_rate = rate(experiment.optimizer.lr, experiment.schedule, step - 1)
            sampled = cadence.sampled(experiment, step)
            size, measurements = self.stepper.step(parts, learning_rate, sampled)
            self.seen += size
            self.last_batch_size = size
            self.trace.append({"step": step, "batch_size": size, **{key: place[key] for key in self.sampler.traced},
                               "learning_rate": learning_rate})
            if sampled:
                with self.clock.diagnosing():
                    self.flush()
                    self.run.record("diagnostics", {"step": step, "batch_size": size, **place,
                                                    "learning_rate": learning_rate, **measurements})
            canonical = cadence.canonical(experiment, step)
            if canonical or cadence.probe(experiment, step):
                with self.clock.diagnosing():
                    self.flush()
                self.clock.pause()
                self.observe(step, probe=not canonical)
                if self.stopped() or cadence.checkpointed(experiment, step):
                    self.save(step)
                if self.stopped():
                    return
                self.clock.resume()

    def finish(self):
        experiment = self.experiment
        memory = {"peak_cuda_allocated_bytes": None, "peak_cuda_reserved_bytes": None}
        if self.device.type == "cuda":
            memory = {"peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(self.device),
                      "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(self.device)}
        step, reason = self.selection.stop if self.stopped() else (experiment.budget.updates, "budget")
        result = {"updates": experiment.budget.updates, "stop": {"step": step, "reason": reason},
                  "training_seconds": self.clock.training, "diagnostic_seconds": self.clock.diagnostics,
                  "wall_seconds": self.clock.wall(), **memory, **self.task.analyze(self.history),
                  "final": self.history[-1]}
        stages = report(self.stepper.optimizer)
        if stages:
            result["optimizer"] = stages
        last = {**{split: self.stepper.evaluate(split) for split in experiment.benchmark.last},
                **self.task.inspect(self.model, experiment.evaluate.batch)}
        if last:
            result["last"] = last
        if self.selection is not None:
            result["best"] = self.select()
        self.run.finish(result)
        return result

    def select(self):
        """Restore the best observation's model, check it, evaluate the final splits on it once, and inspect it."""
        benchmark = self.experiment.benchmark
        if self.best is None:
            raise FloatingPointError(f"No {benchmark.selection} loss of the run is finite")
        self.model.load_state_dict(self.best["model"])
        observed = next(row for row in self.history if row["step"] == self.best["step"])[benchmark.selection]
        again = self.stepper.evaluate(benchmark.selection)
        if again != observed:
            raise RuntimeError(f"The best model evaluates to {again}, not to the observed {observed}")
        return {"step": self.best["step"], benchmark.selection: again,
                **{split: self.stepper.evaluate(split) for split in benchmark.final},
                **self.task.inspect(self.model, self.experiment.evaluate.batch)}
