"""Contracts written before the CUDA compiler adapter and GPU benchmarking."""

import gc
import os
import unittest

import torch
from torch.nn import functional as F

from experiments.gpt_mini import Config
from experiments.optimizer_benchmark.attention import make_model, sparsemax
from experiments.patience_benchmark.registry import make_optimizer
from experiments.compiled_benchmark.backend import compile_model, graph_sparsemax, graph_counts


class ProjectionTests(unittest.TestCase):
    def test_capture_projection_matches_values_jacobian_and_large_translations(self):
        scores = torch.tensor([[0.7, 0.6, -1.0, -float("inf")]], dtype=torch.float64, requires_grad=True)
        torch.testing.assert_close(graph_sparsemax(scores), sparsemax(scores))
        self.assertTrue(torch.autograd.gradcheck(graph_sparsemax, (scores,), eps=1e-6, atol=1e-5))
        for offset in (1000, 10000, 100000):
            torch.testing.assert_close(graph_sparsemax(scores + offset), graph_sparsemax(scores))
        weights = graph_sparsemax(scores)
        torch.testing.assert_close(weights.sum(-1), torch.ones(1, dtype=scores.dtype))
        self.assertEqual(weights[0, -1].item(), 0)

    def test_nonfinite_or_empty_rows_propagate_nonfinite_loss_without_bad_gather(self):
        for row in ([float("nan"), 0], [float("inf"), 0], [-float("inf"), -float("inf")]):
            actual = graph_sparsemax(torch.tensor([row]))
            self.assertTrue(actual.isnan().all().item())


@unittest.skipUnless(os.environ.get("OPTIMIZER_BENCH_CUDA") == "1", "CUDA compile is explicit")
class CompileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        cls.cfg = Config(vocab_size=8, n_layers=1, n_heads=2, d_model=16, d_ff=32, max_seq_len=8)
        cls.tokens = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7], [7, 6, 5, 4, 3, 2, 1, 0]], device="cuda")

    def models(self, attention, method):
        eager = make_model(self.cfg, attention, 0, "cuda")
        compiled = make_model(self.cfg, attention, 0, "cuda")
        compile_model(compiled, attention, method)
        self.assertEqual(list(dict(eager.named_parameters())), list(dict(compiled.named_parameters())))
        self.assertEqual(list(eager.state_dict()), list(compiled.state_dict()))
        self.assertIs(compiled.embed.weight, compiled.unembed.weight)
        return eager, compiled

    def test_compiled_losses_gradients_and_muon_routing_match_eager(self):
        for attention in ("softmax", "sparsemax"):
            eager, compiled = self.models(attention, "magma_muon")
            opts = [make_optimizer("magma_muon", model, 0.01, 0) for model in (eager, compiled)]
            for step in range(3):
                tokens = self.tokens.roll(step, 1)
                losses = []
                for model, opt in zip((eager, compiled), opts):
                    opt.zero_grad(set_to_none=True)
                    loss = F.cross_entropy(model(tokens).flatten(0, 1), tokens.roll(-1, 1).flatten())
                    losses.append(loss.detach().clone())
                    loss.backward()
                torch.testing.assert_close(losses[0], losses[1], rtol=1e-4, atol=1e-5)
                for p, q in zip(eager.parameters(), compiled.parameters()):
                    torch.testing.assert_close(p.grad, q.grad, rtol=3e-4, atol=2e-5)
                for opt in opts:
                    opt.step()
                for p, q in zip(eager.parameters(), compiled.parameters()):
                    torch.testing.assert_close(p, q, rtol=3e-4, atol=2e-5)
            wrapped = opts[1]
            self.assertEqual(len(wrapped.masked_parameters), 4)
            self.assertIn("v", wrapped.base.state[compiled.embed.weight])
            self.assertNotIn("momentum", wrapped.base.state[compiled.embed.weight])
            for opt in opts:
                opt.close()
            del eager, compiled, opts
            gc.collect()
            torch.cuda.empty_cache()

    def test_compiled_fisher_factors_are_fresh_and_match_eager(self):
        for attention in ("softmax", "sparsemax"):
            eager, compiled = self.models(attention, "adafisher")
            opts = [make_optimizer("adafisher", model, 0.0001, 0) for model in (eager, compiled)]
            previous = None
            for step in range(3):
                tokens = self.tokens.roll(step, 1)
                for model, opt in zip((eager, compiled), opts):
                    opt.zero_grad(set_to_none=True)
                    loss = F.cross_entropy(model(tokens).flatten(0, 1), tokens.roll(-1, 1).flatten())
                    loss.backward()
                actual = opts[1].collector.measurements
                self.assertEqual(len(actual), 5)
                for p, q in zip(eager.parameters(), compiled.parameters()):
                    if p in opts[0].collector.measurements:
                        for expected_factor, actual_factor in zip(opts[0].collector.measurements[p], actual[q]):
                            torch.testing.assert_close(actual_factor, expected_factor, rtol=4e-4, atol=2e-5)
                current = actual[compiled.blocks[0].ffn.w_in.weight][0].detach().clone()
                if previous is not None:
                    self.assertFalse(torch.equal(current, previous))
                previous = current
                for opt in opts:
                    opt.step()
            for opt in opts:
                opt.close()
            del eager, compiled, opts
            gc.collect()
            torch.cuda.empty_cache()

    def test_cuda_graphs_are_recorded_and_replayed_across_train_eval(self):
        for attention in ("softmax", "sparsemax"):
            model = make_model(self.cfg, attention, 0, "cuda")
            compile_model(model, attention, "adamw")
            opt = make_optimizer("adamw", model, 0.001, 0)
            for _ in range(6):
                opt.zero_grad(set_to_none=True)
                F.cross_entropy(model(self.tokens).flatten(0, 1), self.tokens.roll(-1, 1).flatten()).backward()
                opt.step()
            torch.cuda.synchronize()
            self.assertGreater(graph_counts()["recorded_graph_nodes"], 0)
            with torch.no_grad():
                first = model(self.tokens).detach().clone()
                second = model(self.tokens[:1]).detach().clone()
                third = model(self.tokens).detach().clone()
            torch.testing.assert_close(first, third)
            torch.testing.assert_close(first[:1], second, rtol=1e-4, atol=1e-5)
            opt.close()
            del opt, model
            gc.collect()
            torch.cuda.empty_cache()
