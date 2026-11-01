"""Bounded diagnostic of task145's CUDA capture failure, after learning closes.

The retained failure occurred on the 96-row tail during the first observation.
This candidate pads evaluation to its declared batch size outside the compiled
graph, marks each inference iteration, and copies valid logits out of captured
storage. Padding is valid for this batch-independent transformer. The probe
does no optimizer updates and never evaluates final test. It is not a retry of
the failed learning case or evidence about generalization.
"""

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time

import torch
from torch.nn import functional as F

from lab.infrastructure.benchmarks import build_task
from lab.infrastructure.engine.loop import evaluate
from lab.infrastructure.engine.residual import ResidualCompiledStepper, ResidualTraining
from lab.infrastructure.loader import load
from lab.infrastructure.store import RunDirectory

HERE = Path(__file__).resolve().parents[1] / "experiments/residual_generalization"


class FixedBatchInference(torch.nn.Module):
    """Keep inference module semantics and pad only the independent batch axis."""

    def __init__(self, compiled, batch):
        super().__init__()
        self.compiled, self.batch = compiled, batch

    def forward(self, inputs):
        size = len(inputs)
        if not 0 < size <= self.batch:
            raise ValueError("Inference batch is outside the declared range")
        padded = F.pad(inputs, (0, 0, 0, self.batch - size))
        torch.compiler.cudagraph_mark_step_begin()
        return self.compiled(padded)[:size].clone()


class FixedBatchStepper(ResidualCompiledStepper):
    def __init__(self, experiment, task, model, clock):
        super().__init__(experiment, task, model, clock)
        self.inference = FixedBatchInference(self.inference, experiment.evaluate.batch)


def main():
    phase = json.loads((HERE / "runs/phase.json").read_text())
    if phase["status"] not in ("learning_closed", "closed"):
        raise ValueError("Keep the frozen learning phase isolated")
    remaining = phase["deadline_monotonic"] - time.monotonic()
    if remaining <= 15:
        raise TimeoutError("The authorized GPU phase budget has expired")
    mode = sys.argv[1]
    if mode not in ("original", "fixed_batch"):
        raise ValueError("Expected original or fixed_batch diagnostic")
    spec = load(HERE).experiments["division-s11-p193-conditioned"]
    stepper = ResidualCompiledStepper if mode == "original" else FixedBatchStepper
    start = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="residual-capture-") as folder:
        trainer = ResidualTraining(spec, RunDirectory(folder), lambda row: None, stepper)
        # Match the original transition: full-state serialization after prepare,
        # followed by the initial observation, with no optimization step.
        trainer.save(0)
        checkpoint_hash = hashlib.sha256((Path(folder) / "checkpoint.pt").read_bytes()).hexdigest()
        saved = copy.deepcopy(trainer.model.state_dict())
        measurements = [trainer.measure() for _ in range(3)]
        for previous, current in zip(measurements, measurements[1:]):
            if previous != current:
                raise AssertionError("Repeated initial observations changed")
        # Warmup/serialization/observations must preserve the original parameters.
        torch.testing.assert_close(trainer.model.state_dict(), saved, rtol=0, atol=0)
        errors = {}
        with torch.no_grad():
            for size in ((1, 48, 96, 512) if mode == "fixed_batch" else (96,)):
                inputs = trainer.task.splits["train"][-size:, :-1]
                expected = trainer.model(inputs).clone()
                actual = trainer.stepper.inference(inputs).clone()
                torch.testing.assert_close(actual, expected, rtol=3e-4, atol=3e-5)
                errors[size] = float((actual - expected).abs().max())
        reference = copy.deepcopy(trainer.model).cpu().eval()
        task = build_task(spec.benchmark, spec.seeds.data, torch.device("cpu"))
        cpu = {name: evaluate(task, reference, task.splits[name], spec.evaluate.batch)
               for name in spec.benchmark.observed}
        report = {"mode": mode, "seconds": time.monotonic() - start,
                  "checkpoint_sha256": checkpoint_hash, "optimizer_updates": 0,
                  "cuda": measurements[-1], "cpu": cpu, "tail_max_logit_error": errors,
                  "cpu_minus_cuda_correct": {
                      name: round((cpu[name]["accuracy"] - measurements[-1][name]["accuracy"]) * len(task.splits[name]))
                      for name in spec.benchmark.observed}}
    (HERE / "runs" / f"capture-{mode}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
