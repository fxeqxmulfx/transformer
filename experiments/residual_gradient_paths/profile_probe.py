"""Disposable CPU compile/gradient checks and throughput; no scored learning."""

import copy
import json
from pathlib import Path
import time

import torch

from experiment import base
from lab.dsl import BranchGates, LinearTransport, swap
from lab.infrastructure.benchmarks import build_task
from lab.infrastructure.engine.gradient_paths import GradientCompiledStepper
from lab.infrastructure.engine.loop import Clock
from lab.infrastructure.nn import build_model


def main():
    torch.set_num_threads(1)
    rows = []
    specs = [BranchGates(depth=8, placement=p, scale=a) for a in (.001, 1., 1000.)
             for p in ("before", "after")]
    specs += [LinearTransport(depth=32, scaling=s) for s in ("ordinary", "depth")]
    for spec in specs:
        experiment = swap(base, "model", spec)
        task = build_task(experiment.benchmark, 1234, "cpu", spec)
        model = build_model(spec, 0, 0)
        reference = copy.deepcopy(model)
        stepper = GradientCompiledStepper(experiment, task, model, Clock(torch.device("cpu")))
        started = time.perf_counter()
        stepper.prepare([64])
        preparation = time.perf_counter() - started
        batch = task.inputs(torch.arange(64))
        expected, targets = task.forward(reference, batch)
        task.loss(expected, targets).backward()
        loss, actual, _ = stepper.update(batch)
        loss.backward()
        gradients = []
        for a, b in zip(model.parameters(), reference.parameters(), strict=True):
            torch.testing.assert_close(a.grad, b.grad, rtol=1e-4, atol=2e-6)
            gradients.append({"compiled": a.grad.detach().tolist(), "eager": b.grad.detach().tolist()})
        torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)
        sampler = task.sampler(64, 1234)
        started = time.perf_counter()
        for _ in range(200):
            parts, _ = sampler.next()
            stepper.step(parts, .01, False)
        seconds = time.perf_counter() - started
        rows.append({"model": vars(spec), "setup_seconds": preparation, "updates_per_second": 200/seconds,
                     "initial_gradients": gradients, "logit_max_error": (actual-expected).abs().max().item()})
        print(json.dumps(rows[-1]), flush=True)
    (Path(__file__).parent / "profile.json").write_text(json.dumps(rows, indent=2)+"\n")


if __name__ == "__main__":
    main()
