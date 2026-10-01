import math
import unittest

import torch

from experiments.optimizer_benchmark.common import DirectionOptimizer, guard_accepts
from experiments.optimizer_benchmark.coordinate import CoordinateOptimizer


class CoordinateTests(unittest.TestCase):
    def parameter(self):
        return torch.nn.Parameter(torch.tensor([1.0, -2.0], dtype=torch.float64))

    def test_sgd_actual_subtraction(self):
        p = self.parameter()
        opt = CoordinateOptimizer([("p", p)], 0.1, rule="sgd")
        p.grad = torch.tensor([2.0, -3.0], dtype=p.dtype)
        opt.step()
        torch.testing.assert_close(p, torch.tensor([0.8, -1.7], dtype=p.dtype))

    def test_adagrad_accumulated_gradient_formula(self):
        p = self.parameter()
        opt = CoordinateOptimizer([("p", p)], 0.1, rule="adagrad", eps=0)
        p.grad = torch.tensor([3.0, 4.0], dtype=p.dtype)
        opt.step()
        p.grad = torch.tensor([0.0, -4.0], dtype=p.dtype)
        opt.step()
        expected = torch.tensor([0.9, -2.1 + 0.1 / math.sqrt(2)], dtype=p.dtype)
        torch.testing.assert_close(p, expected)
        torch.testing.assert_close(opt.state[p]["v"], torch.tensor([4.5, 16.0], dtype=p.dtype))

    def test_amsgrad_retains_the_previous_maximum(self):
        p = self.parameter()
        opt = CoordinateOptimizer([("p", p)], 0.1, beta=0.5, beta2=0.5, eps=0)
        p.grad = torch.tensor([2.0, -2.0], dtype=p.dtype)
        opt.step()
        p.grad.zero_()
        opt.step()
        displacement = 0.15 / math.sqrt(2)
        torch.testing.assert_close(p, torch.tensor([1 - displacement, -2 + displacement], dtype=p.dtype))
        torch.testing.assert_close(opt.state[p]["maximum"], torch.full_like(p, 2))
        torch.testing.assert_close(opt.state[p]["v"], torch.ones_like(p))

    def test_adamx_time_varying_momentum_changes_history(self):
        p = self.parameter()
        opt = CoordinateOptimizer([("p", p)], 0.1, rule="adamx", schedule="inverse",
                                  beta=0.5, beta2=0.5, eps=0)
        p.grad = torch.tensor([2.0, -2.0], dtype=p.dtype)
        opt.step()
        p.grad.zero_()
        opt.step()
        torch.testing.assert_close(opt.state[p]["maximum"], torch.full_like(p, 4.5))
        displacement = 0.1 / math.sqrt(2) + 0.1 / 12
        torch.testing.assert_close(p, torch.tensor([1 - displacement, -2 + displacement], dtype=p.dtype))

    def test_adamnc_second_moment_is_the_true_mean(self):
        p = self.parameter()
        opt = CoordinateOptimizer([("p", p)], 0.01, rule="adamnc", schedule="inverse")
        for values in [[1, 2], [3, 0], [-2, 4]]:
            p.grad = torch.tensor(values, dtype=p.dtype)
            opt.step()
        torch.testing.assert_close(opt.state[p]["v"], torch.tensor([14 / 3, 20 / 3], dtype=p.dtype))

    def test_adamw_agrees_with_independent_torch_implementation(self):
        p, reference = self.parameter(), self.parameter()
        opt = CoordinateOptimizer([("p", p)], 0.003, rule="adamw", decay=0.01)
        native = torch.optim.AdamW([reference], lr=0.003, weight_decay=0.01, foreach=False)
        for values in [[1, -3], [0, 2], [-2, 0]]:
            p.grad = torch.tensor(values, dtype=p.dtype)
            reference.grad = p.grad.clone()
            opt.step()
            native.step()
            torch.testing.assert_close(p, reference, rtol=1e-12, atol=1e-12)

    def test_constant_adamx_equals_amsgrad(self):
        p, q = self.parameter(), self.parameter()
        first = CoordinateOptimizer([("p", p)], 0.001, rule="adamx")
        second = CoordinateOptimizer([("q", q)], 0.001, rule="amsgrad")
        for values in [[2, 1], [-1, 0], [0, 4]]:
            p.grad = torch.tensor(values, dtype=p.dtype)
            q.grad = p.grad.clone()
            first.step()
            second.step()
            torch.testing.assert_close(p, q)

    def test_zero_gradient_does_not_create_nan(self):
        for rule in ["adagrad", "adam", "adamw", "amsgrad", "adamx", "adamnc"]:
            with self.subTest(rule=rule):
                p = self.parameter()
                opt = CoordinateOptimizer([("p", p)], 0.01, rule=rule)
                p.grad = torch.zeros_like(p)
                opt.step()
                torch.testing.assert_close(p, self.parameter())


class GuardTests(unittest.TestCase):
    def test_joint_norm_and_alignment(self):
        g = [torch.tensor([1.0]), torch.tensor([1.0])]
        self.assertTrue(guard_accepts(g, [x * 0.75 for x in g]))
        self.assertFalse(guard_accepts(g, [-x for x in g]))
        self.assertFalse(guard_accepts(g, [g[0] * 0.5, g[1] * 1.5]))
        self.assertFalse(guard_accepts(g, [x * 0.1 for x in g]))

    def test_rejection_keeps_updated_optimizer_state(self):
        class BadCandidate(DirectionOptimizer):
            def propose(self, parameters):
                for p in parameters:
                    self.state[p]["calls"] = self.state[p].get("calls", 0) + 1
                return [-p.grad for p in parameters]

        p = torch.nn.Parameter(torch.tensor([1.0]))
        opt = BadCandidate([("p", p)], 0.5, guarded=True)
        for _ in range(10):
            p.grad = p.detach().clone()  # Actual derivative of x^2 / 2.
            opt.step()
        torch.testing.assert_close(p, torch.tensor([1 / 1024]))
        self.assertEqual(opt.state[p]["calls"], 10)
        self.assertEqual(opt.guard_accepted, 0)

    def test_duplicate_tied_parameters_are_rejected(self):
        p = torch.nn.Parameter(torch.ones(2))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            CoordinateOptimizer([("embedding", p), ("unembedding", p)], 0.1)
