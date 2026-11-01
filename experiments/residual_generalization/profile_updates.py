"""Diagnostic copies only: compile, validate and profile the new residual paths.

No diagnostic checkpoint initializes learning. The phase ledger charges all
CUDA compilation, warmup, profiler overhead and failed attempts to task145.
Use the Python3.15 sampling profiler as a parent, never an attach that needs
ptrace permission. The script writes no learning accuracy or selected model.
"""

import copy
import json
from pathlib import Path
import platform
import statistics
import time

import torch
from torch.profiler import ProfilerActivity, profile

from lab.dsl import swap
from lab.infrastructure.benchmarks import build_task
from lab.infrastructure.engine.residual import ResidualCompiledStepper
from lab.infrastructure.loader import load
from lab.infrastructure.nn import build_model

HERE = Path(__file__).resolve().parent
OUT = HERE / "runs/profiles"


def stamp():
    torch.cuda.synchronize()
    return time.perf_counter()


def advance(stepper, sampler, count):
    start = stamp()
    for _ in range(count):
        parts, _ = sampler.next()
        stepper.step(parts, .003, False)
    elapsed = stamp() - start
    norms = stepper.gradient_norms()
    if not all(0 <= value < float("inf") for value in norms):
        raise FloatingPointError("Nonfinite profiling gradient")
    return elapsed


def main():
    from phase import require_budget
    require_budget()
    torch.set_num_threads(1)
    OUT.mkdir(exist_ok=True, parents=True)
    study = load(HERE)
    labels = ["division-s11-p97-add", "division-s11-p97-qk", "division-s11-p97-conditioned",
              "mechanism-s11-scaled", "mechanism-s11-attnres"]
    reports = []
    for label in labels:
        require_budget()
        spec = study.experiments[label]
        task = build_task(spec.benchmark, spec.seeds.data, torch.device("cuda"))
        model = build_model(spec.model, task.vocab, spec.seeds.model).cuda()
        stepper = ResidualCompiledStepper(spec, task, model, None)
        sampler = task.sampler(spec.budget.batch, spec.seeds.batch_seed)
        saved = copy.deepcopy((model.state_dict(), stepper.state_dict(), sampler.state()))
        inputs = task.splits["train"][:256, :-1]
        with torch.no_grad():
            expected = model(inputs).clone()
        start = stamp()
        stepper.prepare(sampler.sizes())
        compilation = stamp() - start
        model.load_state_dict(saved[0]); stepper.load_state_dict(saved[1])
        with torch.no_grad():
            torch.compiler.cudagraph_mark_step_begin()
            actual = stepper.inference(inputs).clone()
        error = float((expected - actual).abs().max())
        torch.testing.assert_close(actual, expected, rtol=3e-4, atol=3e-5)
        # Validate the actual compiled training gradient before profiling updates.
        batch = task.splits["train"][:256]
        stepper.optimizer.zero_grad(set_to_none=True)
        model.train()
        logits, targets = task.forward(model, batch)
        task.loss(logits, targets).backward()
        expected_gradients = [p.grad.detach().clone() for p in model.parameters()]
        stepper.optimizer.zero_grad(set_to_none=True)
        torch.compiler.cudagraph_mark_step_begin()
        loss, _, _ = stepper.update(batch)
        loss.backward()
        gradient_error = max(float((p.grad - reference).abs().max())
                             for p, reference in zip(model.parameters(), expected_gradients, strict=True))
        for p, reference in zip(model.parameters(), expected_gradients, strict=True):
            torch.testing.assert_close(p.grad, reference, rtol=2e-3, atol=2e-4)
        stepper.optimizer.zero_grad(set_to_none=True)
        warmup = advance(stepper, sampler, 20)
        repeats = [advance(stepper, sampler, 200) for _ in range(3)]
        events = []
        if spec.model.mixing in ("add", "conditioned_qk"):
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as profiler:
                advance(stepper, sampler, 20)
            profiler.export_chrome_trace(str(OUT / f"{spec.model.mixing}-cuda.json"))
            for item in profiler.key_averages():
                events.append({"name": item.key, "calls": item.count,
                    "self_cpu_us": item.self_cpu_time_total, "self_device_us": item.self_device_time_total})
            (OUT / f"{spec.model.mixing}-cuda.txt").write_text(profiler.key_averages().table(
                sort_by="self_device_time_total", row_limit=40))
        report = {"label": label, "mode": spec.model.mixing, "python": platform.python_version(),
            "torch": torch.__version__, "gpu": torch.cuda.get_device_name(),
            "parameters": sum(p.numel() for p in model.parameters()), "compilation_seconds": compilation,
            "eager_compiled_max_logit_error": error, "eager_compiled_max_gradient_error": gradient_error, "warmup20_seconds": warmup,
            "repeat200_seconds": repeats, "updates_per_second": 200 / statistics.median(repeats),
            "events": events, "peak_cuda_bytes": torch.cuda.max_memory_allocated()}
        reports.append(report)
        (OUT / "results.json").write_text(json.dumps(reports, indent=2) + "\n")
        print(json.dumps({k: v for k, v in report.items() if k != "events"}), flush=True)
        del model, stepper, task, saved, inputs, expected, actual, batch, logits, targets, expected_gradients, loss
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
