"""Independent optimality checks for the metric and causal simplex routing."""

import unittest
from unittest.mock import patch

import numpy as np
from scipy.optimize import minimize
import torch

from convex_mqar.certify import encode, train_metric, value_decode
from convex_mqar.convex import ConvexRecall

from .support import batch, recall_from_raw


class ConvexTests(unittest.TestCase):
    def test_binary_encoder_is_injective_at_full_vocabulary(self):
        expected = np.array([[int(bit) for bit in f"{token:013b}"[::-1]]
                             for token in range(8192)])
        np.testing.assert_array_equal(encode(np.arange(8192), 13), expected)
        model = ConvexRecall(8192, np.ones(13))
        np.testing.assert_array_equal(model.codes.numpy(), expected)
        self.assertEqual(len(np.unique(expected, axis=0)), 8192)

    def test_metric_solver_recovers_the_unconstrained_global_minimum(self):
        for bits in (1, 6, 13):
            for seed in (0, 1, 42):
                with self.subTest(bits=bits, seed=seed):
                    w, report = train_metric(bits, np.random.default_rng(seed))
                    np.testing.assert_allclose(w, 1, atol=1e-7)
                    self.assertTrue(np.all(w >= 1))
                    self.assertAlmostEqual(report["objective"], bits / 4, places=7)
                    self.assertLessEqual(report["raw_metric_max_error"], 1e-6)

    def test_projection_satisfies_kkt_conditions_including_masked_slots(self):
        rng = np.random.default_rng(11)
        for size in (1, 2, 9, 129):
            z = torch.tensor(rng.normal(size=(10, size)), dtype=torch.float64)
            if size > 1:
                z[0, 1:] = -torch.inf
                z[1] = 0
                z[2, ::2] = -torch.inf
            a = ConvexRecall.project_simplex(z)
            self.assertTrue(torch.isfinite(a).all())
            self.assertTrue((a >= 0).all())
            torch.testing.assert_close(a.sum(-1), torch.ones(10, dtype=a.dtype))
            for row_z, row_a in zip(z, a):
                active = row_a > 1e-12
                threshold = (row_z[active] - row_a[active]).mean()
                torch.testing.assert_close(row_z[active] - row_a[active],
                                           threshold.expand(active.sum()))
                self.assertTrue((row_z[~active] <= threshold + 1e-12).all())
                self.assertTrue((row_a[~row_z.isfinite()] == 0).all())

    def test_projection_matches_independent_constrained_qp(self):
        rng = np.random.default_rng(7)
        for size in (2, 5, 9):
            for _ in range(3):
                costs = rng.normal(size=size)
                objective = lambda a: np.dot(a, a) / 4 + np.dot(costs, a)
                solved = minimize(objective, np.full(size, 1 / size), method="SLSQP",
                                  bounds=[(0, None)] * size,
                                  constraints={"type": "eq", "fun": lambda a: a.sum() - 1},
                                  options={"ftol": 1e-12, "maxiter": 200})
                self.assertTrue(solved.success, solved.message)
                a = ConvexRecall.project_simplex(torch.tensor(-2 * costs)).numpy()
                np.testing.assert_allclose(a, solved.x, atol=2e-6)
                self.assertAlmostEqual(objective(a), solved.fun, places=9)

    def test_projection_shift_invariance_ties_and_known_two_slot_solution(self):
        z = torch.tensor([[0., 0., 0.], [2., -1., -3.]], dtype=torch.float64)
        a = ConvexRecall.project_simplex(z)
        torch.testing.assert_close(a, torch.tensor([[1/3, 1/3, 1/3], [1., 0., 0.]],
                                                  dtype=torch.float64))
        torch.testing.assert_close(ConvexRecall.project_simplex(z + 101.25), a)
        torch.testing.assert_close(ConvexRecall.project_simplex(torch.tensor([[-0.5, -1.]])),
                                   torch.tensor([[0.75, 0.25]]))

    def test_actual_forward_costs_equal_weighted_bit_mismatches(self):
        model = ConvexRecall(64, np.arange(1, 7))
        tokens, positions, _ = batch(count=3)
        captured = []
        project = model.project_simplex
        with patch.object(model, "project_simplex", side_effect=lambda z: (
                captured.append(z.detach().clone()), project(z))[1]):
            model(tokens, positions)
        scores = captured[0]
        for row in range(len(tokens)):
            for query, position in enumerate(positions[row].tolist()):
                q = tokens[row, position].item()
                for slot, k in enumerate(tokens[row, :8:2].tolist()):
                    expected = sum(r + 1 for r in range(6) if (q ^ k) & (1 << r))
                    self.assertEqual(scores[row, query, slot].item(), -2 * expected)
                self.assertEqual(scores[row, query, -1].item(), -1)

    def test_full_length_recall_matches_the_raw_sequence_oracle(self):
        for length in (64, 128, 256, 512):
            tokens, positions, labels = batch(43, 3, length, 8192)
            model = ConvexRecall(8192, np.ones(13))
            predictions = model(tokens, positions)
            torch.testing.assert_close(predictions, labels)
            expected = [[recall_from_raw(raw, p) for p in query]
                        for raw, query in zip(tokens.tolist(), positions.tolist())]
            torch.testing.assert_close(predictions, torch.tensor(expected))

    def test_first_occurrences_and_unseen_keys_select_null(self):
        tokens, _, _ = batch()
        model = ConvexRecall(64, np.ones(6))
        first = torch.arange(0, 8, 2).expand(len(tokens), -1)
        self.assertTrue((model(tokens, first) == -1).all())
        for row in tokens:
            unused = next(key for key in range(32) if key not in row[:8:2])
            row[15] = unused
        self.assertTrue((model(tokens, torch.full((len(tokens), 1), 15)) == -1).all())

    def test_future_pair_values_are_excluded(self):
        model = ConvexRecall(64, np.ones(6))
        tokens, _, _ = batch(count=3)
        before = tokens.clone()
        tokens[:, 3:] = (tokens[:, 3:] + 5) % 64
        positions = torch.full((3, 1), 2)
        torch.testing.assert_close(model(tokens, positions), model(before, positions))
        self.assertTrue((model(tokens, positions) == -1).all())

    def test_repeated_values_accumulate_mass_before_decoding(self):
        weights = np.array([[0.3, 0.3, 0.4], [0.1, 0.1, 0.2]])
        values = np.array([8, 8, 9])
        np.testing.assert_array_equal(value_decode(weights, values, np.array([0., 0.6])),
                                      [8, -1])
        tokens, positions, _ = batch()
        tokens[:, 1:8:2] = 32
        model = ConvexRecall(64, np.ones(6))
        self.assertTrue((model(tokens, positions) == 32).all())
