"""EXPERIMENT_PLAN.md, step 4: optional QKNorm initialization, with a learned scale."""

import math
import unittest

import torch
from torch.nn import functional as F

from lab.domain.experiment import require_continuation
from lab.dsl import QKNorm, ScaledDot, describe, substitute, swap
from lab.infrastructure.nn import build_model
from lab.infrastructure.nn.attention import QKNormScores

from examples import gptmini, modular


class QKNormInitialScaleTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_scale_is_optional_positive_finite_and_not_a_boolean(self):
        self.assertIsNone(QKNorm().initial_scale)
        for scale in (0, -1, float('inf'), float('-inf'), float('nan'), True, False):
            with self.subTest(scale=scale), self.assertRaises(ValueError):
                QKNorm(initial_scale=scale)
        self.assertEqual(QKNorm(initial_scale=1.0).initial_scale, 1.0)

    def test_old_descriptions_resume_at_the_default_and_reject_a_new_start(self):
        run = modular(gptmini())
        old = describe(run)
        del old['model']['block']['attention']['scores']['initial_scale']
        require_continuation(old, run)
        with self.assertRaisesRegex(ValueError, 'initial_scale'):
            require_continuation(old, swap(run, 'model.block.attention.scores.initial_scale', 1.0))

    def test_default_parameter_retains_original_rounding_and_explicit_one_is_exact(self):
        for head in (8, 16, 32, 64):
            with self.subTest(head=head):
                default = QKNormScores(heads=4, head=head, eps=1e-6)
                self.assertTrue(torch.equal(default.log_alpha, torch.full((4,), .5 * math.log(head))))
                one = QKNormScores(heads=4, head=head, eps=1e-6, initial_scale=1.0)
                self.assertTrue(torch.equal(one.log_alpha, torch.zeros(4)))
                self.assertTrue(torch.equal(one.log_alpha.exp(), torch.ones(4)))
                self.assertTrue(one.log_alpha.requires_grad)

    def test_every_non_score_parameter_remains_identical_across_score_initializations(self):
        for width, depth in ((64, 2), (128, 6)):
            for seed in range(3):
                spec = gptmini(width=width, depth=depth, heads=4)
                pair = [build_model(substitute(spec, QKNorm, scores), vocab=32, seed=seed)
                        for scores in (QKNorm(), QKNorm(initial_scale=1.0), ScaledDot())]
                images = [{name: value for name, value in model.state_dict().items()
                           if not name.endswith('log_alpha')} for model in pair]
                with self.subTest(width=width, depth=depth, seed=seed):
                    self.assertEqual(images[0].keys(), images[1].keys())
                    self.assertEqual(images[0].keys(), images[2].keys())
                    for name in images[0]:
                        self.assertTrue(torch.equal(images[0][name], images[1][name]), name)
                        self.assertTrue(torch.equal(images[0][name], images[2][name]), name)

    def test_scores_and_query_key_gradients_match_normalized_dot_product_and_scale_still_learns(self):
        generator = torch.Generator().manual_seed(109)
        source = [torch.randn(2, 3, 5, 8, generator=generator, dtype=torch.float64) for _ in range(2)]
        upstream = torch.randn(2, 3, 5, 5, generator=generator, dtype=torch.float64)
        queries, keys = [value.clone().requires_grad_() for value in source]
        reference_queries, reference_keys = [value.clone().requires_grad_() for value in source]
        scores = QKNormScores(heads=3, head=8, eps=1e-6, initial_scale=1.0).double()
        actual = scores(queries, keys, None, None)
        expected = F.normalize(reference_queries, dim=-1, eps=1e-6) @ F.normalize(
            reference_keys, dim=-1, eps=1e-6).transpose(-2, -1)
        self.assertTrue(torch.equal(actual, expected))
        actual.backward(upstream)
        expected.backward(upstream)
        torch.testing.assert_close(queries.grad, reference_queries.grad)
        torch.testing.assert_close(keys.grad, reference_keys.grad)
        torch.testing.assert_close(scores.log_alpha.grad, (expected.detach() * upstream).sum((0, 2, 3)))
        self.assertGreater(int(torch.count_nonzero(scores.log_alpha.grad)), 0)


if __name__ == '__main__':
    unittest.main()
