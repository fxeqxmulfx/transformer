# Copyright (c) 2026 Mikhail Borisov
"""Small device and graph checks; no training experiment or GPU benchmark."""

from __future__ import annotations

import unittest

import torch

import lean_optim as opt


@unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
class CudaTests(unittest.TestCase):
    """Numerical checks for cuda."""

    def test_buffered_rules_replay_with_live_gradients_rates_and_state(self) -> None:
        """Compare captured and eager updates after caches have been warmed."""
        factories = {
            "AMSGradW": lambda p, r: opt.AMSGradW(p, rate=r),
            "AdamX": lambda p, r: opt.AdamX(p, rate=r),
            "AdamNC": lambda p, r: opt.AdamNC(p, rate=r),
            "Muon": lambda p, r: opt.Muon(p, rate=r),
            "Dash-NDB": lambda p, r: opt.Dash(p, rate=r, block=2, solver=opt.NewtonDB(3)),
            "Dash-CN": lambda p, r: opt.Dash(p, rate=r, block=2, solver=opt.CoupledNewton(3)),
            "Dash-Chebyshev": lambda p, r: opt.Dash(p, rate=r, block=2, solver=opt.Chebyshev(5, 8)),
            "AMSGradMD": lambda p, r: opt.AMSGradMD(p, rate=r),
            "AMSGradMD-guard": lambda p, r: opt.AMSGradMD(p, rate=r, guard_sigma=0.5),
            "Guarded": lambda p, r: opt.Guarded(opt.AMSGradW(p, rate=r)),
            "Magma": lambda p, r: opt.Magma(opt.AMSGradW(p, rate=r), updates=100),
        }
        for name, factory in factories.items():
            with self.subTest(rule=name):
                p = torch.nn.Parameter(torch.full((2, 3), 0.25, dtype=torch.float64, device="cuda"))
                q = torch.nn.Parameter(p.detach().clone())
                p.grad, q.grad = torch.full_like(p, 0.1), torch.full_like(q, 0.1)
                rate = torch.tensor(0.01, device="cuda")
                captured, eager = factory([p], rate), factory([q], rate)
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(5):
                        captured.step()
                        eager.step()
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    captured.step()
                for gradient, value in ((0.2, 0.02), (0.0, 0.03), (-0.1, 0.01)):
                    p.grad.fill_(gradient)
                    q.grad.fill_(gradient)
                    rate.fill_(value)
                    graph.replay()
                    eager.step()
                    torch.testing.assert_close(p, q, rtol=1e-12, atol=1e-12)

    def test_graph_replay_reads_the_current_rate_and_gradient(self) -> None:
        """Check graph replay reads the current rate and gradient."""
        p = torch.nn.Parameter(torch.ones(2, device="cuda"))
        p.grad = torch.full_like(p, 2.0)
        rate = torch.tensor(0.1, device="cuda")
        optimizer = opt.SGD([p], rate=rate)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            optimizer.step()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            optimizer.step()
        with torch.no_grad():
            p.fill_(1.0)
        graph.replay()
        torch.testing.assert_close(p, torch.full_like(p, 0.8))
        rate.fill_(0.2)
        p.grad.fill_(3.0)
        graph.replay()
        torch.testing.assert_close(p, torch.full_like(p, 0.2))

    def test_coordinate_and_matrix_updates_match_cpu_with_shared_gradients(self) -> None:
        """Check coordinate and matrix updates match cpu with shared gradients."""
        for cls, kwargs in (
            (opt.AMSGradW, {}),
            (opt.AdamX, {}),
            (opt.Muon, {}),
            (opt.Dash, {"block": 2, "solver": opt.CoupledNewton(12)}),
            (opt.ScheduleFreeAdamW, {}),
        ):
            with self.subTest(rule=cls.__name__):
                cpu = torch.nn.Parameter(torch.arange(6, dtype=torch.float64).reshape(2, 3) / 10)
                cuda = torch.nn.Parameter(cpu.detach().cuda())
                reference, actual = cls([cpu], **kwargs), cls([cuda], **kwargs)
                for sign in (1, -1, 0):
                    cpu.grad = sign * torch.arange(1, 7, dtype=torch.float64).reshape(2, 3)
                    cuda.grad = cpu.grad.cuda()
                    reference.step()
                    actual.step()
                    torch.testing.assert_close(cuda.cpu(), cpu, atol=1e-11, rtol=1e-11)


if __name__ == "__main__":
    unittest.main()
