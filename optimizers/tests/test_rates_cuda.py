"""Graph replay and warmed CUDA allocations with nonzero-dimensional rate storage."""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING

import torch

import lean_optim as opt

if TYPE_CHECKING:
    from torch import Tensor
    from torch.optim import Optimizer

_NAMES = ("SGD", "AMSGradMD-guard", "AMSGradMDW", "AdamMDW")


@dataclass
class _Run:
    """A selected matrix and scalar auxiliary parameter sharing a live rate."""

    parameters: list[Tensor]
    rate: Tensor
    optimizer: Optimizer
    scheduler: opt.LearningRateSchedule

    @classmethod
    def create(cls, name: str, shape: tuple[int, ...], dtype: torch.dtype) -> _Run:
        """Start matching histories without relying on a model or training experiment."""
        parameters = [
            torch.nn.Parameter(torch.full(size, 0.25, device="cuda", dtype=dtype))
            for size in ((4, 6), ())
        ]
        for parameter in parameters:
            parameter.grad = torch.full_like(parameter, 0.125)
        rate = torch.full(shape, 0.03125, device="cuda", dtype=dtype)
        options = {"guard_sigma": 0.5} if name.endswith("-guard") else {}
        optimizer = getattr(opt, name.removesuffix("-guard"))(parameters, rate=rate, **options)
        scheduler = opt.LearningRateSchedule(optimizer, warmup=2, total_steps=7, decay="linear")
        return cls(parameters, rate, optimizer, scheduler)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
class CudaRateShapeTests(unittest.TestCase):
    """Keep scalar rate views live without copying device storage or synchronizing steps."""

    def test_graph_replay_reads_shaped_rates_through_warmup_and_decay(self) -> None:
        """Compare scalar and three-dimensional one-element rates on actual CUDA graphs."""
        for name in _NAMES:
            with self.subTest(rule=name):
                captured = _Run.create(name, (1, 1, 1), torch.float64)
                reference = _Run.create(name, (), torch.float64)
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(5):
                        captured.optimizer.step()
                        reference.optimizer.step()
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    captured.optimizer.step()
                for index in range(10):
                    for run in (captured, reference):
                        run.scheduler.step(index)
                        for parameter in run.parameters:
                            parameter.grad.fill_((-1) ** index / 8)
                    graph.replay()
                    reference.optimizer.step()
                    for parameter, expected in zip(
                        captured.parameters, reference.parameters, strict=True
                    ):
                        torch.testing.assert_close(parameter, expected, rtol=0, atol=0)
                self.assertIs(captured.optimizer.param_groups[0]["lr"], captured.rate)
                self.assertEqual(captured.rate.shape, (1, 1, 1))

    def test_shaped_rates_and_schedulers_allocate_no_warmed_cuda_storage(self) -> None:
        """Inspect raw allocator events in eight dtype/rule cases with matrix/scalar weights."""
        for name, dtype in product(_NAMES, (torch.float32, torch.float64)):
            with self.subTest(rule=name, dtype=dtype), tempfile.TemporaryDirectory() as directory:
                run = _Run.create(name, (1, 1, 1), dtype)
                for _ in range(5):
                    run.optimizer.step()
                    run.scheduler.step()
                run.scheduler.step(0)
                torch.cuda.synchronize()
                with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ],
                    profile_memory=True,
                ) as profile:
                    for _ in range(10):
                        run.optimizer.step()
                        run.scheduler.step()
                    torch.cuda.synchronize()
                path = Path(directory) / "trace.json"
                profile.export_chrome_trace(str(path))
                events = json.loads(path.read_text())["traceEvents"]
                self.assertTrue(any(event.get("cat") == "kernel" for event in events))
                allocations = [
                    event["args"]["Bytes"]
                    for event in events
                    if event["name"] == "[memory]"
                    and event["args"]["Device Type"] == 1
                    and event["args"]["Bytes"] > 0
                ]
                self.assertEqual(allocations, [])


if __name__ == "__main__":
    unittest.main()
