"""Loss, gradient, state and update fidelity for the entire compiled step."""

import os
import unittest

import torch
from torch.nn import functional as F

from gpt_mini.gpt_mini import Config
from gpt_mini.infrastructure.benchmark.optimizer_benchmark.attention import make_model
from gpt_mini.infrastructure.benchmark.patience_benchmark.registry import METHODS, make_optimizer
from gpt_mini.infrastructure.benchmark.full_compile_benchmark.step import FullStep, mask_plan
from gpt_mini.infrastructure.benchmark.compiled_benchmark.backend import graph_counts


class MaskTests(unittest.TestCase):
    def test_precomputed_masks_preserve_the_original_generator_stream(self):
        cfg = Config(vocab_size=8, n_layers=1, n_heads=2, d_model=16, d_ff=32, max_seq_len=8)
        for seed in range(3):
            model = make_model(cfg, "softmax", seed)
            opt = make_optimizer("magma_muon", model, .01, seed)
            expected = torch.tensor([opt.draw_masks(4) for _ in range(100)])
            torch.testing.assert_close(mask_plan(100, 4, seed), expected)
            opt.close()


@unittest.skipUnless(os.environ.get("OPTIMIZER_BENCH_CUDA") == "1", "CUDA is explicit")
class FullStepTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        cls.cfg = Config(vocab_size=8, n_layers=1, n_heads=2, d_model=16, d_ff=32, max_seq_len=8)
        cls.tokens = torch.tensor([[0,1,2,3,4,5,6,7], [7,6,5,4,3,2,1,0]], device="cuda")

    def test_all_24_updates_and_states_match_eager_on_both_attentions(self):
        for attention in ("softmax", "sparsemax"):
            for method in METHODS:
                with self.subTest(attention=attention, method=method.name):
                    torch.compiler.reset()
                    eager = make_model(self.cfg, attention, 0, "cuda")
                    model = make_model(self.cfg, attention, 0, "cuda")
                    opt = make_optimizer(method.name, eager, .001, 0)
                    step = FullStep(model, attention, method.name, .001, 0, 2, 8, compiled=True)
                    for index in range(3):
                        inputs = self.tokens.roll(index, 1)
                        targets = inputs.roll(-1, 1)
                        opt.zero_grad(set_to_none=True)
                        loss = F.cross_entropy(eager(inputs).flatten(0,1), targets.flatten())
                        loss.backward()
                        actual = step.batch(inputs, targets)
                        torch.testing.assert_close(actual, loss.detach(), rtol=2e-4, atol=2e-5)
                        for p, q in zip(eager.parameters(), model.parameters()):
                            torch.testing.assert_close(q.grad, p.grad, rtol=1e-3, atol=4e-5)
                        if method.name.startswith("adafisher"):
                            for p, q in zip(eager.parameters(), model.parameters()):
                                if p in opt.collector.measurements:
                                    for a, b in zip(opt.collector.measurements[p], step.factors[q]):
                                        torch.testing.assert_close(a, b, rtol=1e-3, atol=4e-5)
                        opt.step()
                        for p, q in zip(eager.parameters(), model.parameters()):
                            torch.testing.assert_close(q, p, rtol=1e-3, atol=4e-5)
                        step.assert_states_match(opt, rtol=1e-3, atol=4e-5)
                    self.assertEqual(step.compilation_settings["fullgraph"], True)
                    torch.cuda.synchronize()
                    self.assertGreater(graph_counts()["recorded_graph_nodes"], 0)
                    self.assertIs(model.embed.weight, model.unembed.weight)
                    opt.close()
                    step.close()
                    del eager, model, opt, step
                    torch.compiler.reset()
                    torch.cuda.empty_cache()
