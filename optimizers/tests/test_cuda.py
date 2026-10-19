"""Small device and graph checks; no training experiment or GPU benchmark."""

import unittest

import torch

import lean_optim as opt


@unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
class CudaTests(unittest.TestCase):
    def test_graph_replay_reads_the_current_rate_and_gradient(self):
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

    def test_coordinate_and_matrix_updates_match_cpu_with_shared_gradients(self):
        for cls, kwargs in ((opt.AMSGradW, {}), (opt.AdamX, {}), (opt.Muon, {}),
                            (opt.Dash, {"block": 2, "solver": opt.CoupledNewton(12)}),
                            (opt.ScheduleFreeAdamW, {})):
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
