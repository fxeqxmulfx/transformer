"""Regressions for allocation churn, cache lifetime and retained Fisher graphs."""

from __future__ import annotations

import gc
import json
import tempfile
import unittest
import weakref
from pathlib import Path
from typing import TYPE_CHECKING

import torch

import lean_optim as opt

if TYPE_CHECKING:
    from collections.abc import Callable


def _factories() -> dict[str, Callable[[list[torch.Tensor]], torch.optim.Optimizer]]:
    return {
        "SGD": opt.SGD,
        "SGD-decay": lambda p: opt.SGD(p, weight_decay=0.1),
        "Adam": opt.Adam,
        "AMSGradW": lambda p: opt.AMSGradW(p, weight_decay=0.1),
        "AdamWDirection": opt.AdamWDirection,
        "AdamX": opt.AdamX,
        "AdamNC": opt.AdamNC,
        "AdaGrad": opt.AdaGrad,
        "RMSProp": opt.RMSProp,
        "Muon": opt.Muon,
        "Dash-NDB": lambda p: opt.Dash(p, block=8, solver=opt.NewtonDB(3)),
        "Dash-CN": lambda p: opt.Dash(p, block=8, solver=opt.CoupledNewton(3)),
        "Dash-Chebyshev": lambda p: opt.Dash(p, block=8, solver=opt.Chebyshev(5, 8)),
        "AMSGradMD": opt.AMSGradMD,
        "AMSGradMD-guard": lambda p: opt.AMSGradMD(p, guard_sigma=0.5),
        "Guarded": lambda p: opt.Guarded(opt.Adam(p)),
        "Magma": lambda p: opt.Magma(opt.Adam(p), updates=3),
        "Clipped": lambda p: opt.Clipped(opt.Adam(p)),
        "ScheduleFreeSGD": opt.ScheduleFreeSGD,
        "ScheduleFreeAdamW": opt.ScheduleFreeAdamW,
        "ScheduleFreeAMSGradW": opt.ScheduleFreeAMSGradW,
        "ScheduleFreeAMSGradMD": opt.ScheduleFreeAMSGradMD,
        "ClippedOptimistic": opt.ClippedOptimistic,
    }


def _parameters(name: str) -> list[torch.Tensor]:
    value = 0.0 if name == "ClippedOptimistic" else 0.25
    parameters = [torch.nn.Parameter(torch.full(shape, value)) for shape in ((16, 24), (16,))]
    for p in parameters:
        p.grad = torch.linspace(-0.1, 0.2, p.numel()).reshape_as(p)
    return parameters


class AllocationTests(unittest.TestCase):
    """Check actual allocations and ownership beyond the mathematical recurrences."""

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_schedule_free_amsgrad_warmed_cuda_steps_allocate_no_device_storage(self) -> None:
        """Check positive raw CUDA allocator events with active kernel tracing."""
        for factory in (opt.ScheduleFreeAMSGradW, opt.ScheduleFreeAMSGradMD):
            with self.subTest(rule=factory.__name__), tempfile.TemporaryDirectory() as directory:
                parameters = [
                    torch.nn.Parameter(torch.full(shape, 0.25, device="cuda"))
                    for shape in ((64, 96), (64,))
                ]
                for p in parameters:
                    p.grad = torch.linspace(-0.1, 0.2, p.numel(), device="cuda").reshape_as(p)
                optimizer = factory(parameters)
                for _ in range(5):
                    optimizer.step()
                torch.cuda.synchronize()
                with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ],
                    profile_memory=True,
                ) as profile:
                    for _ in range(5):
                        optimizer.step()
                    torch.cuda.synchronize()
                trace = Path(directory) / "trace.json"
                profile.export_chrome_trace(str(trace))
                events = json.loads(trace.read_text())["traceEvents"]
                self.assertTrue(any(event.get("cat") == "kernel" for event in events))
                allocations = [
                    event["args"]["Bytes"]
                    for event in events
                    if event["name"] == "[memory]"
                    and event["args"]["Device Type"] == 1
                    and event["args"]["Bytes"] > 0
                ]
                self.assertEqual(allocations, [])

    def test_warmed_steps_do_not_allocate_parameter_sized_tensors(self) -> None:
        """Check raw allocator events, including temporaries freed within one operator."""
        for name, factory in _factories().items():
            with self.subTest(rule=name), tempfile.TemporaryDirectory() as directory:
                parameters = _parameters(name)
                optimizer = factory(parameters)
                for _ in range(5):
                    optimizer.step()
                with torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU], profile_memory=True
                ) as profile:
                    optimizer.step()
                    optimizer.step()
                trace = Path(directory) / "trace.json"
                profile.export_chrome_trace(str(trace))
                events = json.loads(trace.read_text())["traceEvents"]
                sizes = [
                    e["args"]["Bytes"]
                    for e in events
                    if e["name"] == "[memory]" and e["args"]["Bytes"] > 0
                ]
                smallest_parameter = min(p.numel() * p.element_size() for p in parameters)
                self.assertTrue(all(size < smallest_parameter for size in sizes), sizes)

    def test_replacing_gradients_does_not_grow_caches_or_retain_old_gradients(self) -> None:
        """Check the ordinary set_to_none=True training lifecycle for every buffered rule."""
        for name, factory in _factories().items():
            with self.subTest(rule=name):
                parameters = _parameters(name)
                optimizer = factory(parameters)
                counts = []
                for _ in range(4):
                    references = []
                    for p in parameters:
                        p.grad = torch.full_like(p, 0.125)
                        references.append(weakref.ref(p.grad))
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    gc.collect()
                    self.assertTrue(all(ref() is None for ref in references))
                    counts.append(sum(len(group) for group in optimizer.workspace.buffers.values()))
                self.assertEqual(counts, [counts[0]] * len(counts))

    def test_fisher_retains_each_forward_factor_through_repeated_backward(self) -> None:
        """Check separate forward lifetimes, accumulation and replay of a retained graph."""
        model = torch.nn.Linear(3, 2, bias=False, dtype=torch.float64)
        optimizer = opt.AdaFisher(model)
        first = torch.tensor([[1.0, 2.0, 3.0], [-1.0, 0.5, 2.0]], dtype=torch.float64)
        second = torch.tensor([[2.0, -1.0, 0.0]], dtype=torch.float64)
        one, two = model(first), model(second)
        two.sum().backward(retain_graph=True)
        one.sum().backward(retain_graph=True)
        h, s = optimizer.factors.fresh[model.weight]
        torch.testing.assert_close(h, first.square().sum(0) + second.square().sum(0))
        torch.testing.assert_close(s, torch.full((2,), 3.0, dtype=torch.float64))
        optimizer.zero_grad(set_to_none=False)
        one.sum().backward(retain_graph=True)
        h, s = optimizer.factors.fresh[model.weight]
        torch.testing.assert_close(h, first.square().sum(0))
        torch.testing.assert_close(s, torch.full((2,), 2.0, dtype=torch.float64))
        optimizer.close()

    def test_loading_clears_scratch_and_mutating_config_refreshes_decoded_options(self) -> None:
        """Check checkpoint independence and in-place changes to serialized configuration."""
        p = torch.nn.Parameter(torch.ones(4))
        p.grad = torch.ones_like(p)
        optimizer = opt.AdamX([p], beta1_decay=opt.Geometric(0.8))
        optimizer.step()
        self.assertTrue(optimizer.workspace.buffers)
        checkpoint = optimizer.state_dict()
        self.assertEqual(set(checkpoint), {"state", "param_groups"})
        optimizer.load_state_dict(checkpoint)
        self.assertFalse(optimizer.workspace.buffers)
        original = optimizer.spec.beta1_decay
        self.assertIs(original, optimizer.spec.beta1_decay)
        optimizer.param_groups[0]["beta1_decay"]["ratio"] = 0.7
        self.assertEqual(optimizer.spec.beta1_decay, opt.Geometric(0.7))
        optimizer.step()

    def test_fisher_accumulates_across_autocast_dtypes_with_promotion(self) -> None:
        """Check that cached factors preserve promotion between BF16 and FP32 forwards."""
        model = torch.nn.Linear(3, 2, bias=False)
        optimizer = opt.AdaFisher(model)
        first = torch.tensor([[0.25, 0.5, 0.75], [1.0, -0.5, 0.25]], dtype=torch.bfloat16)
        second = torch.tensor([[0.1, 0.2, 0.3]])
        with torch.autocast("cpu", dtype=torch.bfloat16):
            one = model(first)
        two = model(second)
        first_coefficients = torch.tensor([[0.125, 0.25], [1.0, 0.5]], dtype=torch.bfloat16)
        second_coefficients = torch.tensor([[0.1, 0.2]])
        (one * first_coefficients).sum().backward()
        (two * second_coefficients).sum().backward()
        h, s = optimizer.factors.fresh[model.weight]
        self.assertEqual(h.dtype, torch.float32)
        self.assertEqual(s.dtype, torch.float32)
        torch.testing.assert_close(
            h, first.square().sum(0).float() + second.square().sum(0), rtol=0, atol=0
        )
        torch.testing.assert_close(
            s,
            first_coefficients.square().sum(0).float() + second_coefficients.square().sum(0),
            rtol=0,
            atol=0,
        )
        optimizer.close()


if __name__ == "__main__":
    unittest.main()
