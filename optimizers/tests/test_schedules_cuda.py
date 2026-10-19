"""Live scheduled rates across CUDA graphs, checkpoint migration and allocation traces."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import torch
from _warmup_cases import Run, gradients, parameters

import lean_optim as opt


def _capture(run: Run) -> torch.cuda.CUDAGraph:
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run.optimizer.step()
    return graph


@unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
class CudaScheduleTests(unittest.TestCase):
    """Exercise host-side schedules while captured updates read persistent rate tensors."""

    def test_graph_replay_tracks_zero_warmup_and_cosine_decay(self) -> None:
        """Compare every replay with eager execution through both schedule boundaries."""
        for name in ("MD", "guard", "override", "AdamX", "native"):
            with self.subTest(rule=name):
                captured, eager = Run.create(name, "cuda"), Run.create(name, "cuda")
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(5):
                        captured.step()
                        eager.step()
                torch.cuda.current_stream().wait_stream(stream)
                graph = _capture(captured)
                captured.scheduler.step(0)
                eager.scheduler.step(0)
                for index in range(25):
                    for run in (captured, eager):
                        for p in run.parameters:
                            p.grad.fill_((-1) ** index / 8)
                    before = [p.detach().clone() for p in captured.parameters]
                    graph.replay()
                    eager.optimizer.step()
                    for p, q, old in zip(
                        captured.parameters, eager.parameters, before, strict=True
                    ):
                        torch.testing.assert_close(p, q, rtol=1e-12, atol=1e-12)
                        if index == 0:
                            self.assertTrue(torch.equal(p, old))
                    for run in (captured, eager):
                        self.assertIs(run.optimizer.param_groups[0]["lr"], run.rate)
                        run.scheduler.step()
                    self.assertEqual(
                        captured.scheduler.get_last_lr(), eager.scheduler.get_last_lr()
                    )

    def test_cpu_checkpoint_continues_on_cuda_and_captures_during_warmup(self) -> None:
        """Preserve caller-held rate storage and move custom clocks to the parameter device."""
        for name in ("MD", "guard", "override", "AdamX"):
            with self.subTest(rule=name):
                reference = Run.create(name, "cpu")
                for _ in range(3):
                    reference.step()
                checkpoint = copy.deepcopy(reference.optimizer.state_dict())
                schedule = copy.deepcopy(reference.scheduler.state_dict())
                restored = Run.create(name, "cuda")
                with torch.no_grad():
                    for p, q in zip(reference.parameters, restored.parameters, strict=True):
                        q.copy_(p)
                restored.optimizer.load_state_dict(checkpoint)
                restored.scheduler.load_state_dict(schedule)
                self.assertIs(restored.optimizer.param_groups[0]["lr"], restored.rate)
                if name == "AdamX":
                    for p in restored.parameters:
                        clock = restored.optimizer.state[p]["step"]
                        self.assertEqual(clock.device, p.device)
                        self.assertEqual(clock.dtype, torch.float64)
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    restored.step()
                reference.step()
                torch.cuda.current_stream().wait_stream(stream)
                graph = _capture(restored)
                for _ in range(20):
                    graph.replay()
                    restored.scheduler.step()
                    reference.step()
                    self.assertEqual(
                        restored.scheduler.get_last_lr(), reference.scheduler.get_last_lr()
                    )
                    for p, q in zip(reference.parameters, restored.parameters, strict=True):
                        torch.testing.assert_close(p, q.cpu(), rtol=1e-10, atol=1e-11)

    def test_native_checkpoint_keeps_the_destination_rate_tensor(self) -> None:
        """Reconnect native optimizer rates when optimizer loading replaces its group dict."""
        original, restored = Run.create("native", "cuda"), Run.create("native", "cuda")
        for _ in range(5):
            original.step()
        with torch.no_grad():
            for p, q in zip(original.parameters, restored.parameters, strict=True):
                q.copy_(p)
        restored.optimizer.load_state_dict(copy.deepcopy(original.optimizer.state_dict()))
        restored.scheduler.load_state_dict(copy.deepcopy(original.scheduler.state_dict()))
        self.assertIs(restored.optimizer.param_groups[0]["lr"], restored.rate)
        for _ in range(20):
            original.step()
            restored.step()
            for p, q in zip(original.parameters, restored.parameters, strict=True):
                torch.testing.assert_close(p, q, rtol=0, atol=0)

    def test_warmed_optimizer_and_scheduler_allocate_no_cuda_storage(self) -> None:
        """Inspect positive allocator events with active kernel tracing, including host steps."""
        runs = [
            Run.create(name, "cuda", dtype)
            for dtype in (torch.float32, torch.float64)
            for name in ("MD", "guard", "override", "AdamX")
        ]
        for factory in (opt.ScheduleFreeAMSGradW, opt.ScheduleFreeAMSGradMD):
            values = parameters("cuda", torch.float32)
            optimizer = factory(values, lr=0.125)
            schedule = opt.LearningRateSchedule(optimizer, warmup=8, total_steps=20, decay="cosine")
            runs.append(Run(values, optimizer, schedule, torch.empty((), device="cuda")))
        for run in runs:
            with (
                self.subTest(rule=type(run.optimizer).__name__),
                tempfile.TemporaryDirectory() as directory,
            ):
                gradients(run.optimizer, 0.125)
                for _ in range(5):
                    run.optimizer.step()
                    run.scheduler.step()
                torch.cuda.synchronize()
                with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ],
                    profile_memory=True,
                ) as profile:
                    for _ in range(18):
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
