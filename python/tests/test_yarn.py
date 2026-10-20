"""YaRN's manuscript ramp and the derived probability-retention contract.

References: arXiv:2309.00071v3 Section 3.2; Lean YaRN.Calibration at 4f4bbc61.
Probabilities are checked against independent finite softmax rows and the
theorem's equal-tail extremizer, not frozen outputs of this implementation.
"""

import math
import unittest

import torch

from lab.infrastructure.nn.attention import softmax
from lab.infrastructure.nn.positions import rope_tables
from lab.infrastructure.nn.yarn import retention_logits, yarn_tables


class YaRNFrequencyTests(unittest.TestCase):
    def test_manuscript_ramp_covers_native_middle_and_pi_bands(self):
        cos, sin = yarn_tables(8, 1024, 256, 10_000)
        for i in range(4):
            omega = 10_000 ** (-i / 4)
            count = 256 * omega / (2 * math.pi)
            gamma = min(1, max(0, (count - 1) / 31))
            adjusted = (1 - gamma) * omega / 4 + gamma * omega
            for position in (0, 1, 17, 1023):
                self.assertAlmostEqual(cos[position, i].item(), math.cos(position * adjusted), delta=3e-5)
                self.assertAlmostEqual(sin[position, i].item(), math.sin(position * adjusted), delta=3e-5)
        torch.testing.assert_close(cos.square() + sin.square(), torch.ones_like(cos))

    def test_unextended_tables_are_native_bit_for_bit(self):
        for length in (1, 7, 8):
            actual = yarn_tables(16, length, 8, 10_000)
            expected = rope_tables(16, length, 10_000)
            for a, b in zip(actual, expected):
                self.assertTrue(torch.equal(a, b))


class YaRNTemperatureTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_equal_tail_probabilities_are_retained_at_all_extended_rows(self):
        for gap in (0.25, 1., 5., 10.):
            scores = torch.zeros(9, 9, dtype=torch.float64)
            for row in range(9):
                scores[row, row // 2] = gap
            actual = softmax(retention_logits(scores, 3))
            baseline = torch.softmax(torch.tensor([gap, 0., 0.], dtype=torch.float64), dim=0)[0]
            for row in range(3, 9):
                torch.testing.assert_close(actual[row, row // 2], baseline, rtol=1e-12, atol=1e-14)
            torch.testing.assert_close(actual[:3], softmax(scores)[:3], rtol=0, atol=0)

    def test_arbitrary_competitors_satisfy_the_original_worst_case_bound(self):
        generator = torch.Generator().manual_seed(21)
        scores = torch.randn(2, 3, 9, 9, generator=generator, dtype=torch.float64)
        actual = softmax(retention_logits(scores, 3))
        for batch in range(2):
            for head in range(3):
                for row in range(3, 9):
                    values, indices = scores[batch, head, row, :row + 1].sort(descending=True)
                    gap = (values[0] - values[1]).item()
                    bound = 1 / (1 + 2 * math.exp(-gap))
                    self.assertGreaterEqual(actual[batch, head, row, indices[0]].item() + 1e-14, bound)

    def test_heads_have_independent_measured_gaps_and_causal_competitor_counts(self):
        scores = torch.zeros(2, 2, 7, 7, dtype=torch.float64)
        gaps = torch.tensor([[1., 2.], [3., 4.]], dtype=torch.float64)
        scores[..., 0] = gaps[..., None]
        adjusted = retention_logits(scores, 3)
        for row in range(3, 7):
            expected = gaps + math.log(row / 2)
            torch.testing.assert_close(adjusted[..., row, 0] - adjusted[..., row, 1], expected)

    def test_common_logit_offset_does_not_change_probabilities(self):
        scores = torch.arange(49, dtype=torch.float64).reshape(7, 7) / 5
        expected = softmax(retention_logits(scores, 3))
        for offset in (-10_000., 10_000.):
            actual = softmax(retention_logits(scores + offset, 3))
            torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)

    def test_future_values_never_enter_temperature_or_gradients(self):
        scores = torch.arange(49, dtype=torch.float64).reshape(7, 7).requires_grad_()
        future = torch.ones(7, 7, dtype=torch.bool).triu(1)
        poisoned = scores.masked_fill(future, torch.nan)
        actual = softmax(retention_logits(poisoned, 3))
        expected = softmax(retention_logits(scores, 3))
        torch.testing.assert_close(actual, expected)
        (actual * torch.arange(7)).sum().backward()
        self.assertTrue(torch.isfinite(scores.grad).all())
        self.assertEqual(scores.grad.masked_select(future).count_nonzero().item(), 0)

    def test_ties_and_unresolved_gaps_keep_native_temperature(self):
        for dtype in (torch.float32, torch.float64):
            scores = torch.zeros(5, 5, dtype=dtype)
            scores[-1] = torch.tensor([0., -torch.finfo(dtype).eps / 4, -1., -2., -3.])
            scores.requires_grad_()
            actual = softmax(retention_logits(scores, 3))
            torch.testing.assert_close(actual, softmax(scores), rtol=0, atol=0)
            (actual * torch.arange(5)).sum().backward()
            self.assertTrue(torch.isfinite(scores.grad).all())

    def test_resolved_small_gap_remains_finite_and_obeys_the_bound(self):
        scores = torch.zeros(5, 5, dtype=torch.float64)
        scores[-1] = torch.tensor([0., -1e-10, -0.2, -0.3, -0.4], dtype=torch.float64)
        scores.requires_grad_()
        actual = softmax(retention_logits(scores, 3))
        self.assertTrue(torch.isfinite(actual).all())
        self.assertGreaterEqual(actual[-1, 0].item(), 1 / (1 + 2 * math.exp(-1e-10)))
        (actual * torch.arange(5)).sum().backward()
        self.assertTrue(torch.isfinite(scores.grad).all())

    def test_derivative_includes_adaptive_gap(self):
        scores = (torch.arange(36, dtype=torch.float64).reshape(6, 6) / 3).requires_grad_()
        self.assertTrue(torch.autograd.gradcheck(lambda x: softmax(retention_logits(x, 3)), (scores,)))

    def test_half_precision_is_calibrated_in_float32(self):
        for dtype in (torch.float16, torch.bfloat16):
            scores = torch.eye(7, dtype=dtype, requires_grad=True)
            actual = retention_logits(scores, 3)
            self.assertEqual(actual.dtype, torch.float32)
            probabilities = softmax(actual)
            torch.testing.assert_close(probabilities[-1, -1],
                                       torch.softmax(torch.tensor([1., 0., 0.]), dim=0)[0])
            (probabilities * torch.arange(7)).sum().backward()
            self.assertTrue(torch.isfinite(scores.grad).all())

    def test_short_rows_are_returned_without_arithmetic(self):
        for length in (1, 2, 3):
            scores = torch.randn(length, length)
            self.assertIs(retention_logits(scores, 3), scores)

    def test_invalid_contexts_and_score_shapes_are_rejected(self):
        for context in (1, 0, True, 3.0):
            with self.subTest(context=context), self.assertRaises(ValueError):
                retention_logits(torch.zeros(4, 4), context)
        for scores in (torch.zeros(4), torch.zeros(3, 4), torch.zeros(4, 4, dtype=torch.int64)):
            with self.subTest(shape=scores.shape), self.assertRaises(ValueError):
                retention_logits(scores, 3)


if __name__ == "__main__":
    unittest.main()
