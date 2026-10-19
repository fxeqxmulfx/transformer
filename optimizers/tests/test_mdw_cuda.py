"""MDW CUDA graph replay, checkpoint migration and raw allocation events."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import torch
from _mdw_cases import NAMES, Run


@unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
class MDWCudaTests(unittest.TestCase):
    """Exercise actual CUDA kernels and every rate transition with warmed workspaces."""

    def test_ordinary_mdw_graph_replays_read_live_rates_and_gradients(self) -> None:
        """Match eager updates through warmup, decay and a zero-rate replay."""
        for name in NAMES[:2]:
            with self.subTest(rule=name):
                captured, eager = Run.create(name, "cuda"), Run.create(name, "cuda")
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(5):
                        captured.step()
                        eager.step()
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    captured.optimizer.step()
                captured.scheduler.step(0)
                eager.scheduler.step(0)
                for index in range(25):
                    captured.gradients((-1) ** index / 8)
                    eager.gradients((-1) ** index / 8)
                    old = [p.detach().clone() for p in captured.values]
                    graph.replay()
                    eager.optimizer.step()
                    for p, q, before in zip(captured.values, eager.values, old, strict=True):
                        torch.testing.assert_close(p, q, rtol=1e-12, atol=1e-12)
                        if index == 0:
                            self.assertTrue(torch.equal(p, before))
                    captured.scheduler.step()
                    eager.scheduler.step()
                    self.assertIs(captured.optimizer.param_groups[0]["lr"], captured.rate)

    def test_cpu_checkpoints_continue_on_cuda_for_all_four_variants(self) -> None:
        """Move float64 clocks and nested factor histories onto the destination device."""
        for name in NAMES:
            with self.subTest(rule=name):
                original, restored = Run.create(name), Run.create(name, "cuda")
                for _ in range(5):
                    original.step()
                with torch.no_grad():
                    for p, q in zip(original.values, restored.values, strict=True):
                        q.copy_(p)
                restored.optimizer.load_state_dict(copy.deepcopy(original.optimizer.state_dict()))
                restored.scheduler.load_state_dict(copy.deepcopy(original.scheduler.state_dict()))
                for index in range(17):
                    for run in (original, restored):
                        run.gradients((-1) ** index / 8)
                        run.step()
                    for p, q in zip(original.values, restored.values, strict=True):
                        torch.testing.assert_close(p, q.cpu(), rtol=1e-10, atol=1e-11)

    def test_warmed_steps_and_schedulers_allocate_no_cuda_storage(self) -> None:
        """Check eight dtype/rule scenarios using positive raw allocator events."""
        for name in NAMES:
            for dtype in (torch.float32, torch.float64):
                with (
                    self.subTest(rule=name, dtype=dtype),
                    tempfile.TemporaryDirectory() as directory,
                ):
                    run = Run.create(name, "cuda", dtype, shape=(128, 128))
                    for _ in range(5):
                        run.step()
                    torch.cuda.synchronize()
                    with torch.profiler.profile(
                        activities=[
                            torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA,
                        ],
                        profile_memory=True,
                    ) as profile:
                        for _ in range(18):
                            run.step()
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
                    self.assertTrue(all(bool(p.isfinite().all()) for p in run.values))
