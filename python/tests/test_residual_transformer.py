"""Pinned independent GPT oracle, data isolation, coordinates and continuation."""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from lab.dsl import (Budget, Checkpoint, Eager, Evaluate, Experiment, ModularPairs,
    ResidualPairs, ResidualTransformer, Schedule, ScheduleFreeAdamW, Seeds, describe, swap)
from lab.infrastructure.benchmarks import build_task
from lab.infrastructure.engine import Engine
from lab.infrastructure.engine.loop import Clock
from lab.infrastructure.engine.residual import ResidualEagerStepper
from lab.infrastructure.nn import build_model
from lab.infrastructure.store import RunDirectory

ROOT = Path(__file__).resolve().parents[2]


def recipe(mode="conditioned_qk", updates=8):
    return Experiment(model=ResidualTransformer(8, 2, 2, 8, mixing=mode),
        benchmark=ResidualPairs(7, "division"), optimizer=ScheduleFreeAdamW(lr=.003),
        schedule=Schedule(warmup=2, inclusive=True), budget=Budget(updates, 8),
        seeds=Seeds(11, 11, 11), evaluate=Evaluate(2, 8), checkpoint=Checkpoint(2),
        execution=Eager(device="cpu"))


def train(experiment, path):
    run = RunDirectory(path)
    run.begin(describe(experiment), {}, "test fixture")
    return Engine().train(experiment, run, lambda row: None), run


class ResidualTransformerTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_every_parameter_and_gradient_match_independent_pinned_source(self):
        fixture = json.loads((ROOT / "python/tests/fixtures/residual_transformer.json").read_text())
        self.assertTrue(fixture["source_commit"].startswith("ad76812"))
        for case in fixture["cases"]:
            with self.subTest(mode=case["mode"], clipped=case["clipped"]):
                model = build_model(ResidualTransformer(8, 2, 2, 8, mixing=case["mode"]), 11, 145).double()
                if case["clipped"]:
                    with torch.no_grad():
                        model.blocks[0].sa.c_attn.weight.mul_(1e-14)
                flat = torch.cat([p.detach().flatten() for p in model.parameters()])
                self.assertEqual(hashlib.sha256(flat.numpy().astype("<f8").tobytes()).hexdigest(),
                                 case["parameter_sha256"])
                actual = model(torch.tensor(case["tokens"]))
                torch.testing.assert_close(actual, torch.tensor(case["logits"], dtype=torch.float64),
                                           rtol=1e-10, atol=1e-12)
                actual.square().mean().backward()
                for name, parameter in model.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    torch.testing.assert_close(parameter.grad,
                        torch.tensor(case["gradients"][name], dtype=torch.float64),
                        rtol=1e-9, atol=1e-11, msg=name)

    def test_partition_preserves_old_training_and_never_observes_final_test(self):
        for prime in (7, 97, 193):
            for operation in ("division", "mixed"):
                old = build_task(ModularPairs(prime, operation), 11, torch.device("cpu"))
                new = build_task(ResidualPairs(prime, operation), 11, torch.device("cpu"))
                torch.testing.assert_close(old.splits["train"], new.splits["train"], rtol=0, atol=0)
                torch.testing.assert_close(old.splits["heldout"],
                    torch.cat((new.splits["validation"], new.splits["test"])), rtol=0, atol=0)
                self.assertEqual(new.spec.observed, ("train", "validation"))
                self.assertEqual(new.spec.last, ())
                self.assertEqual(new.spec.final, ())
                self.assertIsNone(new.spec.selection)
                inputs = [tuple(row[:2]) for rows in new.splits.values() for row in rows.tolist()]
                self.assertEqual(len(inputs), len(set(inputs)))

    def test_stable_stop_requires_consecutive_validation_and_training_after_minimum(self):
        spec = ResidualPairs()
        def row(step, train=1., validation=1.):
            return {"step": step, "train": {"accuracy": train}, "validation": {"accuracy": validation}}
        self.assertFalse(spec.stable([row(4000), row(5000), row(6000)]))
        self.assertFalse(spec.stable([row(5000), row(6000, validation=.98), row(7000)]))
        self.assertFalse(spec.stable([row(5000), row(6000, train=.98), row(7000)]))
        self.assertFalse(spec.stable([row(5000), row(6000)]))
        self.assertTrue(spec.stable([row(5000), row(6000), row(7000)]))

    def test_full_state_continuation_and_reserved_test_not_evaluated(self):
        for mode in ("add", "attnres", "qk", "conditioned_qk"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                spec = recipe(mode)
                path = Path(directory)
                _, straight = train(spec, path / "straight")
                train(swap(spec, "budget.updates", 4), path / "continued")
                result, continued = train(spec, path / "continued")
                a, b = straight.checkpoint("cpu"), continued.checkpoint("cpu")
                for key in ("model", "optimizer", "permutation"):
                    torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
                for key in ("cursor", "rotary_random_state", "step", "examples_seen"):
                    self.assertEqual(a[key], b[key])
                self.assertEqual(result["stop"], {"step": 8, "reason": "budget"})
                for row in continued.records("history"):
                    self.assertNotIn("test", row)
                self.assertNotIn("best", result)
                self.assertNotIn("test", result)

    def test_stable_stop_keeps_current_checkpoint_and_survives_resume(self):
        spec = recipe(updates=20)
        spec = swap(spec, "benchmark.target_accuracy", 1e-9)
        spec = swap(spec, "benchmark.minimum_updates", 2)
        with tempfile.TemporaryDirectory() as directory, patch.object(
                ResidualEagerStepper, "evaluate", return_value={"accuracy": 1., "loss": 0.}):
            result, run = train(spec, Path(directory))
            self.assertEqual(result["stop"], {"step": 6, "reason": "stable_validation"})
            before = copy.deepcopy(run.checkpoint("cpu"))
            again, run = train(spec, Path(directory))
            self.assertEqual(again["stop"], result["stop"])
            torch.testing.assert_close(run.checkpoint("cpu")["model"], before["model"], rtol=0, atol=0)
            self.assertEqual(run.checkpoint("cpu")["step"], 6)

    def test_prepare_restores_complete_state_even_if_compilation_changes_it(self):
        from lab.infrastructure.engine.eager import EagerStepper
        spec = recipe()
        task = build_task(spec.benchmark, 11, torch.device("cpu"))
        model = build_model(spec.model, 7, 11)
        stepper = ResidualEagerStepper(spec, task, model, Clock(torch.device("cpu")))
        before = copy.deepcopy((model.state_dict(), stepper.state_dict()))
        def perturb(self, sizes):
            with torch.no_grad():
                self.optimizer.eval()
                self.model.residual_mixers[0].query_scale.mul_(2)
                for parameter in self.model.parameters():
                    parameter.add_(.25)
                self.optimizer.param_groups[0]["step"] += 1
        with patch.object(EagerStepper, "prepare", perturb):
            stepper.prepare([8])
        torch.testing.assert_close((model.state_dict(), stepper.state_dict()), before, rtol=0, atol=0)
        self.assertTrue(model.training)
        # The conditioner must bind restored buffers rather than stale ones.
        stepper.conditioner()

    def test_foreach_optimizer_matches_independent_explicit_xz_recurrence(self):
        from lab.infrastructure.optim.residual_schedule_free import ResidualScheduleFreeAdamW
        torch.manual_seed(145)
        parameters = [torch.nn.Parameter(torch.randn(shape, dtype=torch.float64))
                      for shape in ((3, 4), (4,), ())]
        optimizer = ResidualScheduleFreeAdamW(parameters, betas=(.9, .98), weight_decay=.1)
        x, z = [[p.detach().clone() for p in parameters] for _ in range(2)]
        v = [torch.zeros_like(p) for p in parameters]
        maximum, total = 0., 0.
        for step, rate in enumerate((0., .001, .003, .002, 0., .003), 1):
            optimizer.param_groups[0]["lr"] = rate
            maximum = max(maximum, rate)
            total += maximum**2
            coefficient = maximum**2 / total if total else 0.
            for p, average, base, variance in zip(parameters, x, z, v, strict=True):
                p.grad = torch.randn_like(p)
                query = .9 * average + .1 * base
                variance.mul_(.98).add_(p.grad.square(), alpha=.02)
                direction = p.grad / ((variance / (1 - .98**step)).sqrt() + 1e-8) + .1 * query
                base.add_(direction, alpha=-rate)
                average.add_(base - average, alpha=coefficient)
            optimizer.step()
            for p, average, base, variance in zip(parameters, x, z, v, strict=True):
                torch.testing.assert_close(p, .9 * average + .1 * base, rtol=2e-14, atol=2e-14)
                torch.testing.assert_close(optimizer.state[p]["base"], base, rtol=2e-14, atol=2e-14)
                torch.testing.assert_close(optimizer.state[p]["second"], variance, rtol=2e-14, atol=2e-14)
            optimizer.eval()
            for p, average in zip(parameters, x, strict=True):
                torch.testing.assert_close(p, average, rtol=2e-14, atol=2e-14)
            optimizer.train()

    def test_multiple_query_coordinates_preserve_full_forward_and_clocks(self):
        spec = recipe()
        model = build_model(spec.model, 7, 11).double()
        with torch.no_grad():
            for mixer in model.residual_mixers:
                mixer.query.mul_(1e6)
        task = build_task(spec.benchmark, 11, torch.device("cpu"))
        stepper = ResidualEagerStepper(spec, task, model, Clock(torch.device("cpu")))
        inputs = task.splits["train"][:, :-1]
        before = model(inputs).detach()
        clocks = {k: v for k, v in stepper.optimizer.param_groups[0].items() if k != "params"}
        factors = stepper.conditioner()
        self.assertTrue(bool((factors > 1).all()))
        torch.testing.assert_close(before, model(inputs), rtol=1e-9, atol=1e-11)
        self.assertEqual(clocks, {k: v for k, v in stepper.optimizer.param_groups[0].items() if k != "params"})
        for mixer in model.residual_mixers:
            state = stepper.optimizer.state[mixer.query]
            torch.testing.assert_close(state["base"], mixer.query)
            self.assertLess(float(mixer.query.detach().square().mean().sqrt()), 2.)


if __name__ == "__main__":
    unittest.main()
