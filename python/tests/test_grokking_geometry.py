"""Exact current-logit cleanup certificates and scientific controls."""

from fractions import Fraction
from pathlib import Path
import sys
import unittest

import torch

STUDY = Path(__file__).resolve().parents[2] / "experiments/grokking_internals"
sys.path.insert(0, str(STUDY))
from geometry_certificates import exact_cell, measure


class GeometryTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def arrays(self):
        x = torch.tensor([[[0., 0.]] * 4, [[4., -4.]] * 4, [[-4., 4.]] * 4], dtype=torch.float64)
        targets = torch.tensor([[0] * 4, [0] * 4, [1] * 4], dtype=torch.long)
        selected = torch.tensor([4, 6, 8, 10], dtype=torch.long)
        return x, selected, targets

    def test_exact_arithmetic_agrees_with_fraction_oracle(self):
        rows = [[.25, -2., 4.], [2., .125, -1.], [3., -4., .5]]
        result = exact_cell(rows, 0)
        values = [[Fraction(x) for x in row] for row in rows]
        means = [sum(row[k] for row in values) / len(rows) for k in range(3)]
        grand_mean = sum(means) / 3
        errors = [sum((x - mean - sum(row) / 3 + grand_mean) ** 2
                      for x, mean in zip(row, means)) for row in values]
        margin = means[0] - max(means[1:])
        self.assertEqual(result["margin"], margin)
        self.assertEqual(result["point_energies"], errors)
        self.assertEqual(result["residual_sum"], sum(errors))
        self.assertEqual(result["certified"], [margin > 0 and 2 * e < margin ** 2 for e in errors])

    def test_exact_condition_survives_square_overflow_and_underflow(self):
        for scale in (1., 1e-200, 1e200):
            result = exact_cell([[scale, -scale], [scale, -scale]], 0)
            self.assertEqual(result["certified"], [True, True])
            self.assertEqual(result["residual_sum"], 0)
            wrong = exact_cell([[scale, -scale], [scale, -scale]], 1)
            self.assertEqual(wrong["certified"], [False, False])

    def test_strict_boundary_rejects_tie_and_flat_logits(self):
        result = exact_cell([[0., 0.], [2., -2.]], 0)
        self.assertEqual(result["margin"], 2)
        self.assertEqual(result["point_energies"], [2, 2])
        self.assertEqual(result["certified"], [False, False])
        self.assertEqual(result["strict_correct"], [False, True])
        x, selected, targets = self.arrays()
        output = measure(torch.zeros_like(x), selected, targets)
        self.assertEqual(output["certified_count"], 0)
        self.assertIsNone(output["structural_fraction"])
        self.assertIsNone(output["signed_safety"])
        self.assertFalse(output["structure_protocol_eligible"])

    def test_balanced_wrong_readout_has_perfect_structure_and_no_certificate(self):
        x, selected, targets = self.arrays()
        output = measure(-x, selected, targets)
        self.assertEqual(output["structural_fraction"], 1)
        self.assertTrue(output["structure_protocol_eligible"])
        self.assertEqual(output["raw_nonzero_accuracy"], 0)
        self.assertEqual(output["positive_reference_margin_fraction"], 0)
        self.assertEqual(output["certified_count"], 0)
        self.assertEqual(output["mean_signed_safety"], -1)

    def test_class_bias_is_restored_and_row_shifts_are_removed(self):
        x, selected, targets = self.arrays()
        biased = x + torch.tensor([20., 0.])
        output = measure(biased, selected, targets)
        self.assertEqual(output["structural_fraction"], 1)
        self.assertEqual(output["raw_nonzero_accuracy"], .5)
        self.assertEqual(output["certified_fraction"], .5)
        self.assertEqual(output["certified_incorrect_count"], 0)
        shifts = torch.arange(12, dtype=torch.float64).reshape(3, 4, 1) * 128
        shifted = measure(biased + shifts, selected, targets)
        for key in ("certified_count", "globally_certified_count", "mean_signed_safety"):
            self.assertEqual(output[key], shifted[key])
        self.assertEqual(output["energy"]["residual_sum"], shifted["energy"]["residual_sum"])

    def test_unselected_training_logits_do_not_enter_reference(self):
        x, selected, targets = self.arrays()
        initial = measure(x, selected, targets)
        changed = x.clone()
        changed[:, 1::2] = torch.tensor([900., -900.])
        changed[0] = torch.tensor([-1000., 1000.])
        after = measure(changed, selected, targets)
        self.assertEqual(initial, after)

    def test_nonuniform_masks_keep_absolute_global_energy(self):
        x, _, targets = self.arrays()
        x[1, :3] = torch.tensor([[2., -2.], [4., -4.], [6., -6.]])
        x[2, :2] = torch.tensor([[-1., 1.], [-5., 5.]])
        selected = torch.tensor([4, 5, 6, 8, 9], dtype=torch.long)
        output = measure(x, selected, targets)
        self.assertEqual(output["certified_fraction"], 1)
        self.assertEqual(output["globally_certified_fraction"], 0)
        self.assertEqual(output["energy"]["scalar_coordinates"], 10)
        self.assertAlmostEqual(output["energy"]["residual_sum"], 32)
        self.assertAlmostEqual(output["energy"]["residual_mean"], 3.2)
        self.assertEqual(output["minimum_cell_size"], 2)
        self.assertEqual(output["two_point_cell_coverage"], 1)

    def test_positive_scaling_leaves_safety_and_certificates_unchanged(self):
        x, selected, targets = self.arrays()
        x[1, 0] += torch.tensor([.5, -.5])
        before = measure(x, selected, targets)
        after = measure(x * 2, selected, targets)
        self.assertEqual(before["certified_count"], after["certified_count"])
        self.assertEqual(before["globally_certified_count"], after["globally_certified_count"])
        self.assertAlmostEqual(before["mean_signed_safety"], after["mean_signed_safety"])
        self.assertAlmostEqual(after["energy"]["residual_sum"], 4 * before["energy"]["residual_sum"])

    def test_observation_preserves_tensors_gradients_and_rng(self):
        x, selected, targets = self.arrays()
        x.requires_grad_()
        x.grad = torch.ones_like(x)
        before, gradient = x.clone(), x.grad.clone()
        rng = torch.random.get_rng_state().clone()
        measure(x, selected, targets)
        self.assertTrue(torch.equal(before, x))
        self.assertTrue(torch.equal(gradient, x.grad))
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))

    def test_invalid_selection_and_cell_semantics_are_rejected(self):
        x, selected, targets = self.arrays()
        with self.assertRaisesRegex(ValueError, "unique"):
            measure(x, selected.repeat(2), targets)
        with self.assertRaisesRegex(ValueError, "integer tensors"):
            measure(x, selected.double(), targets)
        targets[1, 1] = 1
        with self.assertRaisesRegex(ValueError, "constant target"):
            measure(x, selected, targets)


if __name__ == "__main__":
    unittest.main()
