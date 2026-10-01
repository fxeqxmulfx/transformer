import unittest
import torch

from experiments.optimizer_benchmark.matrix import DashOptimizer, muon_polar
from experiments.optimizer_benchmark.roots import (
    fitted_power, guarded_scale, inverse_fourth_cn, inverse_fourth_evd, inverse_fourth_ndb,
)


class RootTests(unittest.TestCase):
    def setUp(self):
        # A rotated SPD matrix with eigenvalues 1/4 and 1.
        self.matrix = torch.tensor([[0.625, -0.375], [-0.375, 0.625]], dtype=torch.float64)
        small = 2 ** 0.5
        self.target = torch.tensor([[(small + 1) / 2, (small - 1) / 2],
                                    [(small - 1) / 2, (small + 1) / 2]], dtype=torch.float64)

    def test_all_spectral_roots_match_closed_form(self):
        results = [inverse_fourth_evd(self.matrix), inverse_fourth_ndb(self.matrix, 12),
                   inverse_fourth_cn(self.matrix, 18)]
        for root in results:
            torch.testing.assert_close(root, self.target, rtol=1e-8, atol=1e-8)
            torch.testing.assert_close(self.matrix @ torch.linalg.matrix_power(root, 4),
                                       torch.eye(2, dtype=root.dtype), rtol=1e-8, atol=1e-8)

    def test_missed_top_eigenvalue_triggers_safe_scale(self):
        matrix = torch.diag(torch.tensor([0.01, 1.0], dtype=torch.float64))
        self.assertEqual(guarded_scale(matrix, torch.tensor(0.01)).item(), 2)
        self.assertAlmostEqual(guarded_scale(matrix, torch.tensor(0.8)).item(), 1.6, places=6)
        self.assertEqual(guarded_scale(torch.zeros_like(matrix)).item(), 1)

    def test_root_output_rescaling(self):
        torch.testing.assert_close(inverse_fourth_ndb(16 * self.matrix, 12), self.target / 2)
        torch.testing.assert_close(inverse_fourth_cn(16 * self.matrix, 18), self.target / 2)

    def test_actual_chebyshev_fit_targets_original_regularized_scale(self):
        matrix = torch.diag(torch.tensor([0.5, 1.0], dtype=torch.float64))
        expected = torch.diag(torch.tensor([0.6, 1.1], dtype=torch.float64).pow(-0.25))
        torch.testing.assert_close(fitted_power(matrix, 0.1), expected, rtol=1e-9, atol=1e-9)

    def test_sample_guard_and_polynomial_reproduction(self):
        actual = fitted_power(self.matrix, 0.2, exponent=2, degree=2, samples=1)
        target = torch.linalg.matrix_power(self.matrix + 0.2 * torch.eye(2, dtype=torch.float64), 2)
        torch.testing.assert_close(actual, target, rtol=1e-10, atol=1e-10)

    def test_batch_roots_preserve_independent_scaling(self):
        batch = torch.stack([self.matrix, 16 * self.matrix])
        torch.testing.assert_close(inverse_fourth_ndb(batch, 12), torch.stack([self.target, self.target / 2]))

    def test_increasing_solver_budget_reduces_inverse_residual(self):
        errors = []
        for steps in [1, 3, 6, 12]:
            root = inverse_fourth_ndb(self.matrix, steps)
            errors.append((self.matrix @ torch.linalg.matrix_power(root, 4)
                           - torch.eye(2, dtype=root.dtype)).norm().item())
        self.assertGreater(errors[0], 0.1)
        self.assertLess(errors[-1], 1e-9)
        self.assertTrue(all(b <= a + 1e-12 for a, b in zip(errors, errors[1:])))


class MatrixTests(unittest.TestCase):
    def test_muon_zero_and_positive_scale_invariance(self):
        torch.manual_seed(1)
        matrix = torch.randn(7, 3, dtype=torch.float64)
        torch.testing.assert_close(muon_polar(matrix), muon_polar(12 * matrix))
        torch.testing.assert_close(muon_polar(matrix).T, muon_polar(matrix.T))
        torch.testing.assert_close(muon_polar(torch.zeros_like(matrix)), torch.zeros_like(matrix))
        # The actual five-step approximation is not an exact unit polar factor.
        self.assertGreater(abs(muon_polar(torch.ones(1, 1)).item() - 1), 0.1)

    def test_residual_blocks_cover_every_parameter_and_vector(self):
        p = torch.nn.Parameter(torch.ones(3, 5, dtype=torch.float64))
        q = torch.nn.Parameter(torch.ones(3, dtype=torch.float64))
        p.grad = torch.arange(1, 16, dtype=p.dtype).view_as(p) / 20
        q.grad = torch.tensor([0.1, 0.2, 0.3], dtype=q.dtype)
        opt = DashOptimizer([("p", p), ("q", q)], 0.001, block_size=2, solver="evd")
        opt.step()
        self.assertTrue(bool((p != 1).all()))
        self.assertTrue(bool((q != 1).all()))
        self.assertEqual(sum(len(g["entries"]) * g["shape"][0] * g["shape"][1]
                             for g in opt.groups), p.numel() + q.numel())

    def test_scalar_grafting_cancels_finite_positive_root(self):
        for solver in ["ndb", "cn", "evd", "chebyshev"]:
            with self.subTest(solver=solver):
                p = torch.nn.Parameter(torch.tensor([[1.0]], dtype=torch.float64))
                p.grad = torch.tensor([[0.25]], dtype=p.dtype)
                opt = DashOptimizer([("p", p)], 1, solver=solver, graft_beta=0,
                                    graft_beta2=0, epsilon=0.25, root_steps=1)
                opt.step()
                torch.testing.assert_close(p, torch.tensor([[0.5]], dtype=p.dtype))
