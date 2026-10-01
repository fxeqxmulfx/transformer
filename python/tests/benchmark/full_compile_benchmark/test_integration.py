"""Whole-step window gathering, steady tracing and checkpoint restoration."""

import os
import unittest

import torch

from gpt_mini.gpt_mini import Config
from gpt_mini.infrastructure.benchmark.optimizer_benchmark.attention import make_model, sparsemax
from gpt_mini.infrastructure.benchmark.optimizer_benchmark.data import training_starts
from gpt_mini.infrastructure.benchmark.full_compile_benchmark.model import FunctionalSparsemax
from gpt_mini.infrastructure.benchmark.full_compile_benchmark.step import FullStep
from gpt_mini.infrastructure.benchmark.compiled_benchmark.telemetry import snapshot


class ProjectionTests(unittest.TestCase):
    def test_functional_projection_values_and_reverse_ad_match_the_original(self):
        scores = torch.tensor([[.7,.6,-1.,-float("inf")]], dtype=torch.float64, requires_grad=True)
        weight = torch.tensor([[1.,2.,3.,4.]], dtype=torch.float64)
        for offset in (0, 1000, 10000, 100000):
            expected = sparsemax(scores + offset)
            actual = FunctionalSparsemax.apply(scores + offset)
            torch.testing.assert_close(actual, expected)
        expected_gradient = torch.autograd.grad((sparsemax(scores)*weight).sum(), scores)[0]
        actual_gradient = torch.func.grad(lambda x: (FunctionalSparsemax.apply(x)*weight).sum())(scores)
        torch.testing.assert_close(actual_gradient, expected_gradient)


@unittest.skipUnless(os.environ.get("OPTIMIZER_BENCH_CUDA") == "1", "CUDA is explicit")
class IntegrationTests(unittest.TestCase):
    def test_compiled_windows_validation_restore_and_steady_graphs(self):
        torch.set_num_threads(4)
        cfg = Config(vocab_size=8,n_layers=1,n_heads=2,d_model=16,d_ff=32,max_seq_len=8)
        tokens = (torch.arange(1600) % 8).to("cuda")
        starts = training_starts(1600,8,2,100,0).to("cuda")
        for attention in ("softmax","sparsemax"):
            for method in ("adamw","magma_muon","adafisher","dash_ndb"):
                with self.subTest(attention=attention, method=method):
                    torch.compiler.reset()
                    from torch._dynamo.utils import counters
                    counters.clear()
                    model = make_model(cfg,attention,0,"cuda")
                    step = FullStep(model,attention,method,.001,0,2,100)
                    for _ in range(5): step.train(tokens,starts)
                    for _ in range(2): step.evaluate(tokens[:80],2)
                    warm = snapshot()
                    saved = {name:t.detach().cpu().clone() for name,t in model.state_dict().items()}
                    expected = step.evaluate(tokens[:80],2)
                    for _ in range(20): step.train(tokens,starts)
                    step.evaluate(tokens[:80],2)
                    for _ in range(20): step.train(tokens,starts)
                    step.evaluate(tokens[:80],2)
                    stable = snapshot()
                    self.assertEqual(warm, stable)
                    self.assertEqual(stable["graph_breaks"], {})
                    self.assertGreater(stable["recorded_graph_nodes"],0)
                    model.load_state_dict(saved)
                    self.assertEqual(step.evaluate(tokens[:80],2), expected)
                    step.close()
                    del step,model
                    torch.compiler.reset()
                    torch.cuda.empty_cache()
