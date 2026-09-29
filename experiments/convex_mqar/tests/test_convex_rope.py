"""Independent checks for the new positional convex mechanism and its margin."""

import itertools
import unittest

import numpy as np
from scipy.optimize import minimize
import torch

from convex_mqar.convex_rope import RopeConvexRecall
from convex_mqar.rope import RotaryAttention

from .support import batch, recall_from_raw


class ConvexRopeTests(unittest.TestCase):
    def test_rotation_matches_the_transformer_at_actual_token_positions(self):
        model = RopeConvexRecall(8192, np.ones(7))
        attention = RotaryAttention(64, 1, 10_000.)
        torch.testing.assert_close(model.inverse_frequency, attention.inverse_frequency)
        tokens, positions, _ = batch(19, 3, 512, 8192)
        rotated = attention.rotate(model.codes[tokens][:, None]).squeeze(1)
        expected = rotated.gather(1, positions[..., None].expand(-1, -1, 64))
        actual = model.rotate(model.codes[tokens.gather(1, positions)], positions)
        torch.testing.assert_close(actual, expected)
        code = model.codes[torch.tensor([123, 123])]
        differently_rotated = model.rotate(code, torch.tensor([0, 511]))
        self.assertGreater((differently_rotated[0]-differently_rotated[1]).square().sum(), 0.1)

    def test_codes_are_injective_and_use_only_the_declared_slow_pairs(self):
        model = RopeConvexRecall(8192, np.ones(7))
        self.assertEqual(torch.unique(model.codes, dim=0).shape[0], 8192)
        self.assertTrue((model.codes[:, :50] == 0).all())
        self.assertTrue((model.codes[:, -1] == 1).all())
        for token in (0, 1, 4095, 4096, 8191):
            expected = [2*((token >> r) & 1)-1 for r in range(13)]
            self.assertEqual(model.codes[token, 50:63].tolist(), expected)

    def test_costs_equal_direct_rotated_distance_and_are_linear_in_pair_weights(self):
        model = RopeConvexRecall(64, np.ones(3))
        tokens, positions, _ = batch(23, 2)
        q = model.rotate(model.codes[tokens.gather(1, positions)], positions).double()
        k = model.rotate(model.codes[tokens[:, :8:2]], torch.arange(0, 8, 2)[None]).double()
        pair_costs = (q[:, :, None]-k[:, None]).square().reshape(2, 4, 4, 32, 2).sum(-1) / 4
        for weights in (torch.tensor([1., 2., 3.]), torch.tensor([3., 1., 2.])):
            full_weights = torch.zeros(32)
            full_weights[-3:] = weights
            model.coordinate_weights.copy_(full_weights.repeat_interleave(2))
            expected = (pair_costs*full_weights).sum(-1)
            torch.testing.assert_close(model.costs(tokens, positions).double(), expected,
                                       atol=2e-6, rtol=1e-6)

    def test_pairwise_trigonometric_bounds_cover_all_sign_patterns_and_extreme_gaps(self):
        model = RopeConvexRecall(8192, np.ones(7))
        signs = torch.tensor(list(itertools.product((-1., 1.), repeat=2)), dtype=torch.float64)
        frequencies = model.inverse_frequency[-7:].double()
        for gap in (-511, -256, -1, 0, 1, 256, 511):
            angle = gap * frequencies
            for i, query in enumerate(signs):
                rotated = torch.stack((query[0]*angle.cos()-query[1]*angle.sin(),
                                       query[0]*angle.sin()+query[1]*angle.cos()), -1)
                for j, key in enumerate(signs):
                    cost = (rotated-key).square().sum(-1) / 4
                    if i == j:
                        self.assertTrue((cost <= angle.square()/2 + 1e-14).all())
                    else:
                        self.assertTrue((cost >= 1-angle.abs()-1e-14).all())
        margin = model.margin
        self.assertLess(margin["match_cost_upper"], 0.165)
        self.assertGreater(margin["mismatch_cost_lower"], 0.616)
        self.assertGreater(margin["match_null_gap"], margin["required_gap"])
        self.assertGreater(margin["null_mismatch_gap"], margin["required_gap"])

    def test_all_benchmark_lengths_recall_the_raw_sequence_oracle(self):
        model = RopeConvexRecall(8192, np.ones(7))
        for length in (64, 128, 256, 512):
            tokens, positions, labels = batch(47, 3, length, 8192)
            torch.testing.assert_close(model(tokens, positions), labels)
            expected = [[recall_from_raw(raw, p) for p in query]
                        for raw, query in zip(tokens.tolist(), positions.tolist())]
            torch.testing.assert_close(model(tokens, positions), torch.tensor(expected))
            weights = model.routing_weights(tokens, positions)
            self.assertTrue((weights.amax(-1) == 1).all())
            torch.testing.assert_close(weights.sum(-1), torch.ones_like(positions).float())

    def test_first_occurrences_unseen_keys_and_repeated_values(self):
        model = RopeConvexRecall(64, np.ones(3))
        tokens, positions, _ = batch()
        first = torch.arange(0, 8, 2).expand(len(tokens), -1)
        self.assertTrue((model(tokens, first) == -1).all())
        tokens[:, 1:8:2] = 32
        self.assertTrue((model(tokens, positions) == 32).all())
        for row in tokens:
            unused = next(key for key in range(32) if key not in row[:8:2])
            row[15] = unused
        unknown_positions = torch.full((len(tokens), 1), 15)
        self.assertTrue((model(tokens, unknown_positions) == -1).all())
        weights = model.routing_weights(tokens, first)
        self.assertTrue((weights[..., -1] == 1).all())

    def test_future_values_do_not_affect_outputs(self):
        model = RopeConvexRecall(64, np.ones(3))
        tokens, _, _ = batch(count=3)
        before = tokens.clone()
        tokens[:, 3:] = (tokens[:, 3:]+5) % 64
        positions = torch.full((3, 1), 2)
        torch.testing.assert_close(model(tokens, positions), model(before, positions))

    def test_routing_matches_an_independent_quadratic_program(self):
        model = RopeConvexRecall(64, np.ones(3))
        tokens, positions, _ = batch(count=1)
        cost = model.costs(tokens, positions)[0, 0].numpy()
        cost = np.append(cost, model.null_cost)
        objective = lambda a: model.quadratic*np.dot(a, a)+np.dot(cost, a)
        solved = minimize(objective, np.full(5, 0.2), method="SLSQP",
                          bounds=[(0, None)]*5,
                          constraints={"type": "eq", "fun": lambda a: a.sum()-1},
                          options={"ftol": 1e-12, "maxiter": 200})
        self.assertTrue(solved.success, solved.message)
        actual = model.routing_weights(tokens, positions)[0, 0].numpy()
        np.testing.assert_allclose(actual, solved.x, atol=1e-6)
        self.assertAlmostEqual(objective(actual), solved.fun, places=8)

    def test_rejects_unsupported_margin_domains_and_uncalibrated_weights(self):
        invalid = ({"width": 32}, {"base": 2.}, {"max_length": 2048},
                   {"width": 63}, {"base": float("nan")}, {"max_length": 0})
        for settings in invalid:
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                RopeConvexRecall(8192, np.ones(7), **settings)
        with self.assertRaises(ValueError):
            RopeConvexRecall(8192, np.zeros(7))
        model = RopeConvexRecall(64, np.ones(3), max_length=16)
        tokens, positions, _ = batch(length=32)
        with self.assertRaises(ValueError):
            model(tokens, positions)
