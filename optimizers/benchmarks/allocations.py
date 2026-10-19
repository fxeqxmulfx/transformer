"""Measure optimizer-only allocations after initialization and warmup.

The tensors and dense gradients are allocated before profiling. This measures
operator allocations, including caching-allocator traffic, rather than changes
in CUDA's reserved pool. Scalene can run the identical workload for line-level
Python/native memory sampling with --seconds.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from _atomic_workload import cases as atomic_cases
from torch import Tensor

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from types import ModuleType


@dataclass
class Workload:
    """An optimizer and its fixed-gradient or full-training update workload."""

    name: str
    optimizer: object
    params: list[Tensor]
    update: Callable[[], object]


def training_update(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer, inputs: Tensor
) -> Callable[[], object]:
    """Construct a training cycle with persistent gradient buffers after warmup."""

    def update() -> None:
        optimizer.zero_grad(set_to_none=False)
        model(inputs).square().mean().backward()
        optimizer.step()

    return update


def cases(device: str, optim: ModuleType) -> Iterator[Workload]:
    """Construct fixed matrix/vector workloads with preallocated dense gradients."""
    factories = {
        "SGD": optim.SGD,
        "SGD-decay": lambda p: optim.SGD(p, weight_decay=0.01),
        "Adam": optim.Adam,
        "AMSGrad": optim.AMSGrad,
        "AMSGradW": lambda p: optim.AMSGradW(p, weight_decay=0.01),
        "AdamWDirection": optim.AdamWDirection,
        "AdamX": optim.AdamX,
        "AdamNC": optim.AdamNC,
        "AdaGrad": optim.AdaGrad,
        "RMSProp": optim.RMSProp,
        "Muon": optim.Muon,
        "Dash-NDB": lambda p: optim.Dash(p, block=32, solver=optim.NewtonDB(steps=3)),
        "Dash-CN": lambda p: optim.Dash(p, block=32, solver=optim.CoupledNewton(steps=3)),
        "Dash-EVD": lambda p: optim.Dash(p, block=32, solver=optim.EVD()),
        "Dash-Chebyshev": lambda p: optim.Dash(
            p, block=32, solver=optim.Chebyshev(degree=5, samples=8)
        ),
        "AMSGradMD": optim.AMSGradMD,
        "AMSGradMD-guard": lambda p: optim.AMSGradMD(p, guard_sigma=0.5),
        "Guarded-Adam": lambda p: optim.Guarded(optim.Adam(p)),
        "Magma-Adam": lambda p: optim.Magma(optim.Adam(p), updates=100),
        "Clipped-Adam": lambda p: optim.Clipped(optim.Adam(p)),
        "ScheduleFreeSGD": optim.ScheduleFreeSGD,
        "ScheduleFreeAdamW": optim.ScheduleFreeAdamW,
        "ClippedOptimistic": optim.ClippedOptimistic,
        "torch-SGD": torch.optim.SGD,
        "torch-AdamW": torch.optim.AdamW,
    }
    for name, factory in factories.items():
        # Mixed matrix/vector parameters exercise the auxiliary paths too.
        initial = 0.0 if name == "ClippedOptimistic" else 0.25
        params = [
            torch.nn.Parameter(torch.full(shape, initial, device=device))
            for shape in ((64, 96), (64,))
        ]
        for p in params:
            p.grad = torch.linspace(-0.1, 0.2, p.numel(), device=device).reshape_as(p)
        optimizer = factory(params)
        yield Workload(name, optimizer, params, optimizer.step)
    model = torch.nn.Linear(96, 64, bias=False, device=device)
    model.weight.grad = torch.full_like(model.weight, 0.1)
    optimizer = optim.AdaFisher(model)
    yield Workload("AdaFisher", optimizer, list(model.parameters()), optimizer.step)
    inputs = torch.linspace(-1, 2, 32 * 96, device=device).reshape(32, 96)
    yield Workload(
        "AdaFisher-training",
        optimizer,
        list(model.parameters()),
        training_update(model, optimizer, inputs),
    )
    control = torch.nn.Linear(96, 64, bias=False, device=device)
    control_optimizer = torch.optim.SGD(control.parameters(), lr=1e-3)
    yield Workload(
        "torch-SGD-training",
        control_optimizer,
        list(control.parameters()),
        training_update(control, control_optimizer, inputs),
    )
    for name, atomic, parameters, update in atomic_cases(device, optim):
        yield Workload(name, atomic, parameters, update)


def tensor_bytes(value: object) -> int:
    """Count unique owned tensor storages in nested state and workspace values."""
    storages = {}
    visited = set()

    def visit(item: object) -> None:
        if id(item) in visited:
            return
        visited.add(id(item))
        if isinstance(item, Tensor):
            storage = item.untyped_storage()
            storages[item.device, storage.data_ptr()] = storage.nbytes()
        elif isinstance(item, dict):
            for child in item.values():
                visit(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
        elif type(item).__module__.startswith("lean_optim"):
            visit(vars(item))

    visit(value)
    return sum(storages.values())


def allocation_totals(
    trace: dict[str, object], device_type: int, iterations: int
) -> dict[str, float]:
    """Count raw allocator events, including allocations freed inside one operator."""
    memory = sorted(
        (
            e
            for e in trace["traceEvents"]
            if e["name"] == "[memory]" and e["args"]["Device Type"] == device_type
        ),
        key=lambda e: e["ts"],
    )
    allocations = [e["args"]["Bytes"] for e in memory if e["args"]["Bytes"] > 0]
    live, peak = 0, 0
    for event in memory:
        live += event["args"]["Bytes"]
        peak = max(peak, live)
    return {
        "allocation_events_per_step": len(allocations) / iterations,
        "bytes_per_step": sum(allocations) / iterations,
        "peak_extra_bytes": peak,
    }


def measure(
    workload: Workload, device: str, trace_path: Path, iterations: int = 5
) -> dict[str, float | int | str]:
    """Measure raw allocator events after initializing and warming the optimizer."""
    for _ in range(5):
        workload.update()
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        resident = torch.cuda.memory_allocated()
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    start = time.perf_counter()
    with torch.profiler.profile(activities=activities, profile_memory=True) as profile:
        for _ in range(iterations):
            workload.update()
        if device == "cuda":
            torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    peak = torch.cuda.max_memory_allocated() - resident if device == "cuda" else 0
    profile.export_chrome_trace(str(trace_path))
    trace = json.loads(trace_path.read_text())
    result = {
        "iterations": iterations,
        "parameter_bytes": sum(p.numel() * p.element_size() for p in workload.params),
        "parameter_dtypes": ",".join(sorted({str(p.dtype) for p in workload.params})),
        "state_bytes": tensor_bytes(getattr(workload.optimizer, "state", {})),
        "workspace_bytes": tensor_bytes(
            (
                getattr(workload.optimizer, "workspace", None),
                getattr(workload.optimizer, "root", None),
                getattr(getattr(workload.optimizer, "factors", None), "workspace", None),
            )
        ),
        "profiled_ms_per_step": elapsed * 1000 / iterations,
    }
    result.update(
        {f"cpu_{key}": value for key, value in allocation_totals(trace, 0, iterations).items()}
    )
    result.update(
        {f"device_{key}": value for key, value in allocation_totals(trace, 1, iterations).items()}
    )
    if device == "cuda":
        result["resident_bytes"] = resident
        result["peak_extra_bytes"] = peak
    return result


def main() -> None:
    """Write reproducible warmed-step allocation measurements as JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=0)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--source-label", default="working-tree")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    if args.source is not None:
        sys.path.insert(0, str(args.source))
    optim = importlib.import_module("lean_optim")
    source = Path(optim.__file__).parent
    digest = hashlib.sha256()
    for file in sorted(source.glob("*.py")):
        digest.update(file.name.encode())
        digest.update(file.read_bytes())
    results = {
        "torch": torch.__version__,
        "device": args.device,
        "default_dtype": "float32",
        "source": args.source_label,
        "source_sha256": digest.hexdigest(),
        "cases": {},
    }
    workloads = list(cases(args.device, optim))
    for workload in workloads:
        name = workload.name
        trace = args.output.with_name(f"{args.output.stem}.{name}.trace.json")
        results["cases"][name] = measure(workload, args.device, trace)
        sys.stdout.write(f"{name} {json.dumps(results['cases'][name])}\n")
        sys.stdout.flush()
    deadline = time.monotonic() + args.seconds
    while time.monotonic() < deadline:
        for workload in workloads:
            workload.update()
    for workload in workloads:
        close = getattr(workload.optimizer, "close", None)
        if close is not None:
            close()
    args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
