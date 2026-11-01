"""Independent finite-epsilon and residual-transport oracles, not fitted labels."""

import copy
from decimal import Decimal, localcontext
import math
import tempfile
import unittest
from unittest.mock import patch

import torch

from lab.dsl import (AccurateBranchGates, BranchGates, LinearTransport, GradientTargets, Experiment, ScheduleFreeAdamW,
                     Schedule, Budget, Seeds, Evaluate, Checkpoint, Eager, MaxAutotune, Diagnostics, describe, swap)
from lab.infrastructure.benchmarks import build_task
from lab.infrastructure.engine.gradient_paths import GradientEagerStepper, GradientCompiledStepper
from lab.infrastructure.engine.loop import Training
from lab.infrastructure.nn import build_model
from lab.infrastructure.store import RunDirectory


def recipe(model=BranchGates()):
    return Experiment(model, GradientTargets(64, 32, 32), ScheduleFreeAdamW(.01, betas=(.9, .98)),
                      Schedule(), Budget(8, 16), Seeds(data=11, batches=11), Evaluate(4, 32),
                      Eager(device="cpu"), checkpoint=Checkpoint(4), diagnostics=Diagnostics(gradients=True))


def equal_state(test, a, b):
    if isinstance(a, torch.Tensor):
        test.assertTrue(torch.equal(a, b))
    elif isinstance(a, dict):
        test.assertEqual(a.keys(), b.keys())
        for key in a:
            equal_state(test, a[key], b[key])
    elif isinstance(a, (tuple, list)):
        test.assertEqual(len(a), len(b))
        for x, y in zip(a, b, strict=True):
            equal_state(test, x, y)
    else:
        test.assertEqual(a, b)


class GradientPathTests(unittest.TestCase):
    def test_scalar_derivative_against_decimal_and_finite_difference(self):
        for scale in (.001, 1., 1000.):
            with localcontext() as ctx:
                ctx.prec = 70
                a, eps = Decimal(str(scale)), Decimal(2) ** -23
                root = (a*a + eps).sqrt()
                derivatives = {"before": float(a * eps / (2 * root**3)),
                               "after": float(a / (2 * root))}
            for placement, expected in derivatives.items():
                spec = BranchGates(placement=placement, scale=scale)
                model = build_model(spec, 0, 0).double()
                inputs = torch.ones(1, 2, 32, dtype=torch.float64)
                value = model(inputs)[0, 0]
                derivative, = torch.autograd.grad(value, model.logits)
                self.assertAlmostEqual(derivative.item() / expected, 1., delta=.015)
                if scale <= .001 or placement == "after":
                    with torch.no_grad():
                        model.logits.fill_(1e-4); upper = model(inputs)[0, 0].item()
                        model.logits.fill_(-1e-4); lower = model(inputs)[0, 0].item()
                    self.assertAlmostEqual((upper-lower) / (2e-4) / expected, 1., delta=1e-6)

    def test_both_placements_represent_teacher_and_start_identically(self):
        for depth in (1, 8):
            for scale in (.001, 1., 1000.):
                initial = []
                for placement in ("before", "after"):
                    spec = BranchGates(depth=depth, scale=scale, placement=placement)
                    task = build_task(GradientTargets(16, 16, 16), 11, "cpu", spec)
                    model = build_model(spec, 0, 0).double()
                    batch = task.splits["test"].double()
                    initial.append(model(batch[:, :-1]))
                    gains = []
                    for index in range(depth):
                        c = (1, 2, 3)[index % 3] / 4
                        gain = (math.sqrt(spec.eps)*c/(scale*math.sqrt(1-c*c)) if placement == "before"
                                else c*math.sqrt(scale*scale+spec.eps)/scale)
                        self.assertTrue(0 < gain < 2)
                        gains.append(math.log(gain/(2-gain)))
                    with torch.no_grad():
                        model.logits.copy_(torch.tensor(gains, dtype=torch.float64))
                    torch.testing.assert_close(model(batch[:, :-1]), batch[:, -1], rtol=1e-12, atol=1e-12)
                torch.testing.assert_close(initial[0], initial[1], rtol=0, atol=0)

    def test_transport_values_and_derivatives_have_independent_closed_form(self):
        for depth in (1, 8, 32):
            for scaling in ("ordinary", "depth"):
                spec = LinearTransport(depth=depth, scaling=scaling)
                model = build_model(spec, 0, 0).double()
                c = -.5 if scaling == "ordinary" else -.5 / depth
                gain = (1+c)**depth
                x = torch.ones(1, 1, 32, dtype=torch.float64)
                with torch.no_grad():
                    model.weight.fill_(1/gain)
                out = model(x)[0, 0]
                grad, = torch.autograd.grad(out, model.weight)
                self.assertAlmostEqual(out.item(), 1., delta=2e-14)
                self.assertAlmostEqual(grad.item()/gain, 1., delta=2e-14)
                if scaling == "depth":
                    self.assertGreaterEqual(gain, .5)
                with torch.no_grad():
                    model.weight.fill_(.25/gain)
                self.assertAlmostEqual(model(x)[0, 0].item(), .25, delta=2e-14)

    def test_targets_and_splits_do_not_come_from_model(self):
        for model_spec in (BranchGates(depth=8), LinearTransport(depth=32)):
            spec = GradientTargets()
            task = build_task(spec, 17, "cpu", model_spec)
            seen = set()
            for split in task.splits.values():
                for row in split:
                    key = tuple(row[:-1].flatten().tolist())
                    self.assertNotIn(key, seen)
                    seen.add(key)
                    expected = (row[0]/4 if isinstance(model_spec, LinearTransport) else
                                row[0] + sum(row[i+1] * ((i%3+1)/4) for i in range(model_spec.depth)))
                    torch.testing.assert_close(expected, row[-1], rtol=0, atol=0)
            self.assertEqual(task.hashes, build_task(spec, 17, "cpu", model_spec).hashes)
            self.assertNotEqual(task.hashes, build_task(spec, 11, "cpu", model_spec).hashes)

    def test_complete_state_resume_and_sealed_test(self):
        self.check_resume(Eager(device="cpu"), GradientEagerStepper)

    def test_compiled_complete_state_resume_and_sealed_test(self):
        self.check_resume(MaxAutotune(device="cpu"), GradientCompiledStepper)

    def test_accurate_compiled_complete_state_resume_and_sealed_test(self):
        self.check_resume(MaxAutotune(device="cpu"), GradientCompiledStepper,
                          AccurateBranchGates(depth=8, placement="before"))

    def check_resume(self, execution, stepper, model=BranchGates(depth=8)):
        def train(path, spec):
            run = RunDirectory(path)
            run.begin(describe(spec), {}, "independent resume test")
            trainer = Training(spec, run, lambda row: None, stepper)
            evaluate = trainer.stepper.evaluate
            def guarded(name):
                self.assertNotEqual(name, "test")
                return evaluate(name)
            trainer.stepper.evaluate = guarded
            trainer()
            return run, trainer

        with tempfile.TemporaryDirectory() as directory:
            spec = swap(recipe(model), "execution", execution)
            uninterrupted, _ = train(directory+'/whole', spec)
            train(directory+'/resumed', swap(spec, "budget.updates", 4))
            resumed, trainer = train(directory+'/resumed', spec)
            clocks = {"training_seconds", "diagnostic_seconds", "wall_seconds"}
            for key in uninterrupted.checkpoint("cpu").keys() - clocks:
                equal_state(self, uninterrupted.checkpoint("cpu")[key], resumed.checkpoint("cpu")[key])
            state = copy.deepcopy((trainer.model.state_dict(), trainer.stepper.state_dict()))
            from lab.infrastructure.engine.eager import EagerStepper
            from lab.infrastructure.engine.compiled import CompiledStepper
            def perturbed(stepper, sizes):
                stepper.optimizer.eval()
                with torch.no_grad():
                    for parameter in stepper.model.parameters():
                        parameter.add_(.125)
                    stepper.optimizer.param_groups[0]["step"] += 10
            parent = EagerStepper if stepper is GradientEagerStepper else CompiledStepper
            with patch.object(parent, "prepare", perturbed):
                trainer.stepper.prepare([16])
            equal_state(self, state, (trainer.model.state_dict(), trainer.stepper.state_dict()))
            self.assertEqual(uninterrupted.records("gradients"), resumed.records("gradients"))


if __name__ == "__main__":
    unittest.main()
