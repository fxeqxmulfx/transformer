"""QKNorm follow-up: raw query/key norms, with an independently checked learned gain."""

import math
import unittest

import torch
from torch.nn import functional as F

from lab.dsl import FusedQKV, LearnedScaledDot, PerHeadQKV, QKNorm, ScaledDot, Softmax, Sparsemax, substitute, swap
from lab.infrastructure.nn import build_model
from lab.infrastructure.nn.attention import LearnedScaledDotScores, softmax

from examples import gptmini


class LearnedScaledDotTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_start_requires_a_positive_finite_number(self):
        for value in (0, -1, math.inf, -math.inf, math.nan, True, False):
            with self.subTest(value=value), self.assertRaises(ValueError):
                LearnedScaledDot(initial_scale=value)

    def test_one_retains_scaled_dot_scores_and_other_model_parameters(self):
        for projections in (FusedQKV(), PerHeadQKV()):
            spec = swap(gptmini(32, 2, 4), 'block.attention.projections', projections)
            models = [build_model(substitute(spec, QKNorm, scores), vocab=17, seed=13)
                      for scores in (ScaledDot(), LearnedScaledDot(), LearnedScaledDot(3.0), QKNorm(initial_scale=1.0))]
            images = [{name: value for name, value in model.state_dict().items()
                       if not name.endswith('log_alpha')} for model in models]
            for image in images[1:]:
                self.assertEqual(image.keys(), images[0].keys())
                for name, value in image.items():
                    self.assertTrue(torch.equal(value, images[0][name]), name)
            tokens = torch.randint(17, (2, 12), generator=torch.Generator().manual_seed(14))
            for weights in (Softmax(), Sparsemax(), Softmax(fused=True)):
                pair = [build_model(substitute(substitute(spec, QKNorm, scores), Softmax, weights),
                                    vocab=17, seed=13) for scores in (ScaledDot(), LearnedScaledDot())]
                actual = [model(tokens) for model in pair]
                self.assertTrue(torch.equal(*actual))
                for logits in actual:
                    logits.square().sum().backward()
                gradients = [{name: value.grad for name, value in model.named_parameters()
                              if not name.endswith('log_alpha')} for model in pair]
                for name, value in gradients[0].items():
                    self.assertTrue(torch.equal(value, gradients[1][name]), name)

    def test_raw_dot_score_gradients_and_head_gain_match_analytic_derivatives(self):
        generator = torch.Generator().manual_seed(31)
        for index in (None, 1):
            shape = (2, 3, 5, 8) if index is None else (2, 5, 8)
            q, k = [torch.randn(shape, dtype=torch.float64, generator=generator, requires_grad=True)
                    for _ in range(2)]
            module = LearnedScaledDotScores(heads=3, head=8, initial_scale=1.0).double()
            with torch.no_grad():
                module.log_alpha.copy_(torch.tensor([-.4, .3, .8], dtype=torch.float64))
            scores = module(q, k, None, index)
            upstream = torch.randn(scores.shape, dtype=torch.float64, generator=generator)
            scores.backward(upstream)
            gain = module.log_alpha.detach().exp()
            factor = (gain.view(1, -1, 1, 1) if index is None else gain[index]) / math.sqrt(8)
            torch.testing.assert_close(q.grad, factor * (upstream @ k.detach()))
            torch.testing.assert_close(k.grad, factor * (upstream.transpose(-2, -1) @ q.detach()))
            expected_gain = torch.zeros(3, dtype=torch.float64)
            if index is None:
                expected_gain = (scores.detach() * upstream).sum((0, 2, 3))
            else:
                expected_gain[index] = (scores.detach() * upstream).sum()
            torch.testing.assert_close(module.log_alpha.grad, expected_gain)
            self.assertGreater(float(expected_gain.abs().sum()), 0)
            self.assertGreater(float((q.detach() * q.grad).sum(-1).abs().sum()), 0)

    def test_fused_softmax_matches_explicit_causal_attention_and_all_gradients(self):
        generator = torch.Generator().manual_seed(53)
        for index in (None, 1):
            shape = (2, 3, 5, 8) if index is None else (2, 5, 8)
            source = [torch.randn(shape, dtype=torch.float64, generator=generator) for _ in range(3)]
            groups = [[tensor.clone().requires_grad_() for tensor in source] for _ in range(2)]
            modules = [LearnedScaledDotScores(3, 8, 2.3).double() for _ in range(2)]
            q, k, v = groups[0]
            explicit = softmax(modules[0](q, k, None, index)) @ v
            fused = modules[1].attend(*groups[1], None, index)
            upstream = torch.randn(explicit.shape, dtype=torch.float64, generator=generator)
            explicit.backward(upstream)
            fused.backward(upstream)
            torch.testing.assert_close(fused, explicit)
            for expected, actual in zip(groups[0], groups[1], strict=True):
                torch.testing.assert_close(actual.grad, expected.grad)
            torch.testing.assert_close(modules[1].log_alpha.grad, modules[0].log_alpha.grad)

    def test_fullgraph_compilation_preserves_sparsemax_model_and_every_gradient(self):
        spec = substitute(substitute(gptmini(32, 2, 4), QKNorm, LearnedScaledDot(3.0)), Softmax, Sparsemax())
        eager, compiled_model = (build_model(spec, vocab=17, seed=61) for _ in range(2))
        compiled = torch.compile(compiled_model, dynamic=False, fullgraph=True)
        generator = torch.Generator().manual_seed(62)
        tokens, targets = [torch.randint(17, (2, 12), generator=generator) for _ in range(2)]
        expected, actual = eager(tokens), compiled(tokens)
        for logits in (expected, actual):
            F.cross_entropy(logits.flatten(0, 1), targets.flatten()).backward()
        torch.testing.assert_close(actual, expected)
        for (name, value), (other_name, other) in zip(eager.named_parameters(), compiled_model.named_parameters(), strict=True):
            self.assertEqual(name, other_name)
            torch.testing.assert_close(other.grad, value.grad)


if __name__ == '__main__':
    unittest.main()
