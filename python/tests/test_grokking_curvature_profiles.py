"""CE/HVP controls independent of the native step and its approximations."""

import copy
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch

from lab.domain.spec import swap
from lab.domain.training import rate
from lab.infrastructure.benchmarks.modular import ModularTask
from lab.infrastructure.engine.eager import EagerStepper
from lab.infrastructure.loader import load
from lab.infrastructure.nn import build_model

STUDY = Path(__file__).resolve().parents[2] / "experiments/grokking_internals"
sys.path.insert(0, str(STUDY))
from curvature_profiles import interpolate, measure, point
from momentum_directions import measure as momentum


class QuarticModel(torch.nn.Module):
    def __init__(self, coefficient, initial=0.):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([initial], dtype=torch.float64))
        self.unused = torch.nn.Parameter(torch.tensor([7.], dtype=torch.float64))
        self.coefficient = coefficient

    def forward(self, rows):
        state = self.weight * rows[:, 0]
        score = state - self.coefficient * state ** 4
        return torch.stack([score, torch.zeros_like(score)], -1)[:, None].expand(-1, 2, -1)


class BinaryTask:
    loss = staticmethod(ModularTask.loss)
    position_losses = staticmethod(ModularTask.position_losses)

    @staticmethod
    def forward(model, rows):
        return model(rows), torch.zeros((len(rows), 2), dtype=torch.long)


class CurvatureControls(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.task = BinaryTask()
        self.rows = torch.ones(5, 1, dtype=torch.float64)
        self.direction = [torch.tensor([-1.], dtype=torch.float64), torch.tensor([9.], dtype=torch.float64)]

    def test_actual_CE_same_initial_loss_slope_and_Hessian_despite_higher_order_term(self):
        for coefficient in (0., 2e9):
            result = point(QuarticModel(coefficient), self.task, self.rows, self.direction, "full", batch=3)
            self.assertAlmostEqual(result["loss"], math.log(2), places=12)
            self.assertAlmostEqual(result["rate_derivative"], -.5, places=12)
            self.assertAlmostEqual(result["rate_curvature"], .25, places=12)
            self.assertAlmostEqual(result["gradient_l2"], .5, places=12)

    def test_weighted_tails_match_independent_feature_mean_oracle(self):
        rows = torch.tensor([[1.], [2.], [4.], [8.], [16.]], dtype=torch.float64)
        result = point(QuarticModel(.1), self.task, rows, self.direction, "answer", batch=3)
        self.assertAlmostEqual(result["rate_derivative"], -float(rows.mean()) / 2, places=12)
        self.assertAlmostEqual(result["rate_curvature"], float(rows.square().mean()) / 4, places=12)
        self.assertAlmostEqual(result["gradient_l2"], float(rows.mean()) / 2, places=12)

    def test_same_initial_quadratic_prediction_does_not_certify_finite_CE_decrease(self):
        eta = .001
        for coefficient, increases in ((0., False), (2 / eta ** 3, True)):
            model = QuarticModel(coefficient)
            initial = point(model, self.task, self.rows, self.direction, "full")
            predicted = eta * initial["rate_derivative"] + eta ** 2 * initial["rate_curvature"] / 2
            self.assertLess(predicted, 0)
            interpolate(model, [torch.tensor([0.]), torch.tensor([7.])],
                         [torch.tensor([eta], dtype=torch.float64), torch.tensor([7.])], 1.)
            after = point(model, self.task, self.rows, self.direction, "full")
            self.assertEqual(after["loss"] > initial["loss"], increases)

    def test_HVP_agrees_with_independent_central_loss_differences(self):
        model = QuarticModel(.1, initial=.3)
        result = point(model, self.task, self.rows, self.direction, "full")
        values = []
        h = 1e-3
        for delta in (-h, 0., h):
            with torch.no_grad():
                model.weight.fill_(.3 + delta)
                output, target = self.task.forward(model, self.rows)
                values.append(float(self.task.loss(output, target)))
        self.assertAlmostEqual(result["rate_derivative"], (values[2] - values[0]) / (2 * h), places=6)
        self.assertAlmostEqual(result["rate_curvature"], (values[2] - 2 * values[1] + values[0]) / h ** 2, places=6)


class NativeCurvatureControls(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        spec = dict(load(STUDY).select([]))["gptmini-seed1"]
        for path, value in (("model.width", 16), ("model.block.attention.heads", 2),
                            ("benchmark.prime", 5), ("budget.batch", 3), ("schedule.warmup", 0)):
            spec = swap(spec, path, value)
        self.spec = spec
        self.task = ModularTask(spec.benchmark, spec.seeds.data, torch.device("cpu"))
        self.model = build_model(spec.model, self.task.vocab, spec.seeds.model).double()
        sampler = self.task.sampler(spec.budget.batch, spec.seeds.batch_seed)
        eager = EagerStepper(spec, self.task, self.model, None)
        for step in range(2):
            parts, _ = sampler.next()
            eager.step(parts, rate(spec.optimizer.lr, spec.schedule, step), False)
        self.snapshot = copy.deepcopy({"step": 2, "model": self.model.state_dict(),
                                       "optimizer": eager.state_dict(), **sampler.state()})

    def equal(self, left, right):
        if isinstance(left, torch.Tensor):
            self.assertTrue(torch.equal(left, right))
        elif isinstance(left, dict):
            self.assertEqual(set(left), set(right))
            for key in left:
                self.equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.equal(a, b)
        else:
            self.assertEqual(left, right)

    def test_real_block_endpoints_and_initial_slopes_match_frozen_native_observer(self):
        expected = momentum(self.model, self.task, self.spec, self.snapshot, batch=3)
        observed = measure(self.model, self.task, self.spec, self.snapshot, batch=3)
        self.equal(expected["minibatch"]["indices_sha256"], observed["minibatch"]["indices_sha256"])
        for name, kind in (("train_all", "full"), ("heldout_nonzero", "answer")):
            records = observed["populations"][name]["records"]
            self.assertAlmostEqual(records[0]["loss"], expected["populations"][name]["before"]["losses"][kind], places=12)
            self.assertAlmostEqual(records[-1]["loss"], expected["populations"][name]["after"]["losses"][kind], places=12)
            self.assertAlmostEqual(records[0]["rate_derivative"],
                expected["populations"][name]["alignment"][kind]["finite_CPU_displacement"]["loss_rate_derivative"], places=10)
            self.assertEqual(sum(r["rate_curvature"] is not None for r in records), 5)

    def test_preserves_weights_moments_gradients_modes_rng_and_failure_cleanup(self):
        self.model.train()
        self.model.blocks[0].eval()
        modes = [module.training for module in self.model.modules()]
        state, snapshot = copy.deepcopy(self.model.state_dict()), copy.deepcopy(self.snapshot)
        for p in self.model.parameters():
            p.grad = torch.ones_like(p)
        rng = torch.random.get_rng_state().clone()
        measure(self.model, self.task, self.spec, self.snapshot, batch=3)

        def fail(*args):
            torch.rand(1)
            raise RuntimeError("curvature failure")

        with patch.object(self.task, "forward", fail), self.assertRaisesRegex(RuntimeError, "curvature failure"):
            measure(self.model, self.task, self.spec, self.snapshot)
        self.equal(state, self.model.state_dict())
        self.equal(snapshot, self.snapshot)
        self.assertEqual(modes, [module.training for module in self.model.modules()])
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        self.assertTrue(all(torch.equal(p.grad, torch.ones_like(p)) for p in self.model.parameters()))

    def test_missing_initial_states_and_nonfinite_directions_are_rejected(self):
        snapshot = copy.deepcopy(self.snapshot)
        snapshot["step"] = 0
        with self.assertRaisesRegex(ValueError, "noninitial"):
            measure(self.model, self.task, self.spec, snapshot)
        model = QuarticModel(0.)
        with self.assertRaises(FloatingPointError):
            point(model, BinaryTask(), torch.ones(3, 1), [torch.tensor([float("nan")]), torch.tensor([1.])], "full")


if __name__ == "__main__":
    unittest.main()
