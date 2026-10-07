"""Independent controls of the common fixed real parameter map.

Source: Structured.TensorGain at c12df24. Compare every learned field,
actual stack inference, both complete likelihoods and their gradients
with an independently scaled original model, not just a trained witness.
"""

import itertools
import math
import unittest

import torch

from lab.dsl import TensorGain, TensorStack
from lab.domain.generative import Parity
from lab.domain.tasks import MQAR
from lab.infrastructure.benchmarks.synthetic.tensor import complete_labels
from lab.infrastructure.nn import build_model
from lab.infrastructure.nn.tensor_heads import scaled_potential


class TensorGainTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_actual_initial_potentials_and_parameter_count_are_preserved(self):
        for width, depth, vocab, context in ((64, 2, 36, 128), (128, 6, 548, 64)):
            spec = TensorStack(width, depth, context)
            original = build_model(spec, vocab, 7)
            gained = build_model(TensorGain(width, depth, context, gain=8.0), vocab, 7)
            torch.testing.assert_close(original.embed.weight, gained.embed.weight, rtol=0, atol=0)
            for name in ("initial", "emission", "branch", "chronology", "relative"):
                torch.testing.assert_close(getattr(original.heads, name),
                                           scaled_potential(getattr(gained.heads, name), 8), rtol=0, atol=0)
            torch.testing.assert_close(original.absolute, gained.absolute * 8, rtol=0, atol=0)
            self.assertEqual(sum(p.numel() for p in gained.parameters()), spec.parameter_count(vocab))
            self.assertEqual(gained.integer_function([1, 9, 10]), original.integer_function([1, 9, 10]))
            self.assertEqual(gained.integer_function([-1, 10]), [-1, 10, 0])

    def test_actual_function_likelihood_and_all_gradients_match_scaled_original(self):
        recall = [1, *itertools.chain.from_iterable((36 + k, 292 + k) for k in range(8)), 36]
        cases = ((64, 2, 68, 19, Parity(), [1, 22, 18, 25], 17, 0),
                 (128, 6, 548, 64, MQAR(symbols=256, pairs=8, queries=8), recall, 292, 1))
        for width, depth, vocab, context, task, raw, answer, branch in cases:
            spec = TensorStack(width, depth, context)
            original = build_model(spec, vocab, 3).double()
            gained = build_model(TensorGain(width, depth, context, gain=8.0), vocab, 3).double()
            with torch.no_grad():
                for parameter in gained.parameters():
                    parameter.normal_(std=.05)
                original.load_state_dict(gained.state_dict())
                for parameter in original.parameters():
                    parameter.mul_(8)
            tokens = torch.tensor([raw])
            targets = torch.full_like(tokens, -100)
            targets[0, -1] = answer
            labels = complete_labels(tokens, targets, task, vocab)
            torch.testing.assert_close(gained(tokens), original(tokens), rtol=1e-10, atol=1e-10)
            lg = gained.complete_losses(tokens, labels, branch)[0, -1]
            lo = original.complete_losses(tokens, labels, branch)[0, -1]
            torch.testing.assert_close(lg, lo, rtol=1e-11, atol=1e-11)
            gg = torch.autograd.grad(lg, tuple(gained.parameters()))
            go = torch.autograd.grad(lo, tuple(original.parameters()))
            for first, second in zip(gg, go, strict=True):
                torch.testing.assert_close(first, second * 8, rtol=1e-8, atol=1e-10)

    def test_actual_complete_objective_remains_convex_in_all_gained_coordinates(self):
        model = build_model(TensorGain(64, 3, 19, gain=8), 68, 0).double()
        tokens = torch.tensor([[1, 22, 21, 18, 25]])
        targets = torch.tensor([[-100, -100, -100, 25, 17]])
        labels = complete_labels(tokens, targets, Parity(), 68)
        parameters = tuple(model.parameters())
        first = [p.detach().clone() for p in parameters]
        second = [torch.randn_like(p) * .2 for p in parameters]

        def loss(fraction):
            with torch.no_grad():
                for parameter, left, right in zip(parameters, first, second, strict=True):
                    parameter.copy_(fraction * left + (1 - fraction) * right)
                return float(model.complete_losses(tokens, labels, 0)[0, -1])

        middle, left, right = loss(.37), loss(1), loss(0)
        self.assertLessEqual(middle, .37 * left + .63 * right + 1e-10)

    def test_nonpositive_or_nonfinite_fixed_gain_is_rejected(self):
        for gain in (0, -8, math.inf, math.nan):
            with self.assertRaises(ValueError):
                TensorGain(64, 2, 19, gain=gain).check()


if __name__ == "__main__":
    unittest.main()
