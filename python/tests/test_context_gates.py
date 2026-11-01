"""Independent conditional optima, derivatives and identifiability for task158."""

import math
import unittest

import torch

from lab.dsl import ContextGates, ContextTargets, GradientTargets, swap
from lab.infrastructure.benchmarks import build_task
from lab.infrastructure.nn import build_model
from test_gradient_paths import recipe


class ContextGateTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_both_finite_optima_and_matched_initial_functions(self):
        for depth in (1, 8):
            for scale in (.001, 1., 1000.):
                initial = []
                for placement in ("before", "after"):
                    spec = ContextGates(depth=depth, scale=scale, placement=placement)
                    task = build_task(ContextTargets(16, 16, 16), 11, "cpu", spec)
                    model = build_model(spec, 0, 0).double()
                    batch = task.splits["test"].double()
                    initial.append(model(batch[:, :-1]))
                    logits = []
                    for condition in task.teacher:
                        row = []
                        for c in condition.tolist():
                            gain = (math.sqrt(spec.eps)*c/(scale*math.sqrt(1-c*c))
                                    if placement == "before" else c*math.sqrt(scale*scale+spec.eps)/scale)
                            self.assertTrue(0 < gain < 2)
                            row.append(math.log(gain/(2-gain)))
                        logits.append(row)
                    negative, positive = torch.tensor(logits, dtype=torch.float64)
                    weights = torch.stack(((negative+positive)/2, (positive-negative)/2), -1)
                    with torch.no_grad():
                        model.weight.copy_(weights)
                    torch.testing.assert_close(model(batch[:, :-1]), batch[:, -1], rtol=1e-12, atol=1e-12)
                self.assertTrue(torch.equal(*initial))

    def test_parameter_gradient_matches_independent_scalar_chain(self):
        for scale in (.001, 1., 1000.):
            for placement in ("before", "after"):
                spec = ContextGates(scale=scale, placement=placement)
                model = build_model(spec, 0, 0)
                inputs = torch.ones(2, 2, 32)
                inputs[0, 0, 0] = -1
                target = inputs[:, 0] + torch.tensor([.25, .75])[:, None] * inputs[:, 1]
                loss = (model(inputs)-target).square().mean()
                gradient, = torch.autograd.grad(loss, model.weight)
                root = math.sqrt(scale*scale+spec.eps)
                amplitude = scale/root
                local = (.5*scale*spec.eps/root**3 if placement == "before" else .5*scale/root)
                expected = torch.tensor([[(2*amplitude-1)*local, -.5*local]])
                torch.testing.assert_close(gradient, expected, rtol=2e-6, atol=1e-20)

    def test_targets_are_conditional_and_disjoint(self):
        spec = ContextGates(depth=8)
        task = build_task(ContextTargets(64, 32, 32), 17, "cpu", spec)
        seen = set()
        for split in task.splits.values():
            self.assertEqual(set(split[:, 0, 0].tolist()), {-1., 1.})
            for row in split:
                key = tuple(row[:-1].flatten().tolist())
                self.assertNotIn(key, seen)
                seen.add(key)
                coefficients = (3, 1, 2) if row[0, 0] > 0 else (1, 2, 3)
                expected = row[0] + sum(row[i+1] * (coefficients[i%3]/4) for i in range(spec.depth))
                torch.testing.assert_close(row[-1], expected, rtol=0, atol=0)

    def test_conditional_model_cannot_silently_use_unconditional_targets(self):
        with self.assertRaisesRegex(ValueError, "must be used together"):
            swap(recipe(ContextGates()), "benchmark", GradientTargets()).check()


if __name__ == "__main__":
    unittest.main()
