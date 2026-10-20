"""Schedule-Free AdamW DSL, scheduled updates and canonical evaluation modes."""

import copy
import math
import unittest
from unittest import mock

import lean_optim
import torch

from lab.dsl import (Budget, Clipped, Compiled, CudaGraph, DenseBranch, Eager, Evaluate,
                     Experiment, Guarded, LabDivision, Magma, ReLU, Rotate90, Schedule,
                     ScheduleFreeAdamW, Seeds, describe, swap)
from lab.domain.optimizers import Optimizer
from lab.domain.spec import blocks_of
from lab.infrastructure.benchmarks import build_task
from lab.infrastructure.engine.compiled import CompiledStepper
from lab.infrastructure.engine.eager import EagerStepper
from lab.infrastructure.nn import build_model
from lab.infrastructure.optim import build_optimizer

from examples import gptmini, modular


def experiment(warmup=0):
    return Experiment(model=Rotate90(8, DenseBranch(ReLU())), benchmark=LabDivision(prime=5),
                      optimizer=ScheduleFreeAdamW(lr=.02, betas=(.6, .75),
                                                 weight_decay=.3, eps=.05, warmup=warmup),
                      schedule=Schedule(warmup=0 if warmup else 2),
                      budget=Budget(updates=6, batch=4), seeds=Seeds(model=0, data=0),
                      evaluate=Evaluate(every=2), execution=Eager(device="cpu"))


class ScheduleFreeIntegrationTests(unittest.TestCase):
    def test_block_is_discoverable_and_rejects_invalid_hyperparameters(self):
        spec = ScheduleFreeAdamW(lr=.01)
        self.assertIn(ScheduleFreeAdamW, blocks_of(Optimizer))
        self.assertEqual(describe(spec)["type"], "ScheduleFreeAdamW")
        for options in ({"lr": 0.}, {"lr": math.inf}, {"betas": (.9, 1.)}, {"betas": (0., .9)},
                        {"betas": (math.nan, .9)}, {"weight_decay": math.inf},
                        {"weight_decay": -.1}, {"eps": math.nan}, {"eps": 0.},
                        {"warmup": -1}, {"warmup": True}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                ScheduleFreeAdamW(**({"lr": .01} | options))

    def test_unsupported_stages_graphs_and_double_warmup_fail_before_training(self):
        base = experiment()
        for stage in (Guarded, Magma):
            with self.subTest(stage=stage.__name__), self.assertRaisesRegex(ValueError, "Schedule-Free"):
                stage(base.optimizer)
        with self.assertRaisesRegex(ValueError, "Schedule-Free"):
            swap(base, "execution", CudaGraph())
        with self.assertRaisesRegex(ValueError, "warmup"):
            swap(base, "optimizer.warmup", 3)
        transformer = swap(modular(gptmini(8, 1, 2), prime=5, updates=4, batch=4, every=2),
                           "optimizer", base.optimizer)
        self.assertIsInstance(swap(transformer, "execution", Compiled()).execution, Compiled)
        self.assertIsInstance(swap(base, "optimizer", Clipped(base.optimizer)).optimizer, Clipped)
        model = torch.nn.Linear(3, 2)
        with self.assertRaisesRegex(ValueError, "eager"):
            build_optimizer(base.optimizer, model, rate=torch.tensor(0.))

    def test_eager_updates_follow_scheduled_package_recurrence_and_resume(self):
        for warmup in (0, 3):
            with self.subTest(warmup=warmup):
                spec = experiment(warmup)
                task = build_task(spec.benchmark, 0, torch.device("cpu"))
                model = build_model(spec.model, task.vocab, 0)
                reference = build_model(spec.model, task.vocab, 0)
                stepper = EagerStepper(spec, task, model, None)
                self.assertIsInstance(stepper.optimizer, lean_optim.ScheduleFreeAdamW)
                self.assertEqual(stepper.optimizer.param_groups[0]["lr"], 0.)
                manual = lean_optim.ScheduleFreeAdamW(reference, lr=.02, betas=(.6, .75),
                                                     weight_decay=.3, eps=.05, warmup=warmup)
                rows = torch.arange(4)
                parts = ((rows, 0, 4),)
                for rate in (0., .005, .02, 0., .01):
                    before = [p.detach().clone() for p in model.parameters()]
                    base = [stepper.optimizer.state[p]["base"].clone() for p in model.parameters()]
                    manual.param_groups[0]["lr"] = rate
                    manual.zero_grad(set_to_none=True)
                    task.loss(*task.forward(reference, task.inputs(rows))).backward()
                    manual.step()
                    stepper.step(parts, rate, False)
                    for p, q in zip(model.parameters(), reference.parameters(), strict=True):
                        self.assertTrue(torch.equal(p, q))
                    if not rate:
                        for p, previous in zip(model.parameters(), base, strict=True):
                            self.assertTrue(torch.equal(stepper.optimizer.state[p]["base"], previous))
                        if not stepper.optimizer.param_groups[0]["maximum"]:
                            for p, previous in zip(model.parameters(), before, strict=True):
                                self.assertTrue(torch.equal(p, previous))
                stepper.optimizer.eval()
                checkpoint = copy.deepcopy(stepper.state_dict())
                resumed_model = build_model(spec.model, task.vocab, 0)
                resumed_model.load_state_dict(copy.deepcopy(model.state_dict()))
                resumed = EagerStepper(spec, task, resumed_model, None)
                resumed.load_state_dict(checkpoint)
                self.assertFalse(resumed.optimizer.param_groups[0]["training"])
                for solver in (stepper, resumed):
                    solver.step(parts, .015, False)
                for p, q in zip(model.parameters(), resumed_model.parameters(), strict=True):
                    self.assertTrue(torch.equal(p, q))

    def test_both_splits_use_averages_and_restore_queries_after_evaluation_errors(self):
        spec = experiment()
        task = build_task(spec.benchmark, 0, torch.device("cpu"))
        model = build_model(spec.model, task.vocab, 0)
        stepper = EagerStepper(spec, task, model, None)
        for rate in (.01, .02, .005):
            stepper.step(((torch.arange(4), 0, 4),), rate, False)
        query = [p.detach().clone() for p in model.parameters()]
        beta = spec.optimizer.betas[0]
        evaluation = [p.detach() + (1 - 1 / beta) * (stepper.optimizer.state[p]["base"] - p.detach())
                      for p in model.parameters()]
        self.assertTrue(any(not torch.equal(p, x)
                            for p, x in zip(model.parameters(), evaluation, strict=True)))

        def observe(task, network, rows, batch):
            self.assertFalse(stepper.optimizer.param_groups[0]["training"])
            for p, x in zip(network.parameters(), evaluation, strict=True):
                torch.testing.assert_close(p, x)
            return {"observations": len(rows)}

        for split in ("train", "heldout"):
            with mock.patch("lab.infrastructure.engine.eager.evaluate", side_effect=observe):
                self.assertGreater(stepper.evaluate(split)["observations"], 0)
            self.assertTrue(stepper.optimizer.param_groups[0]["training"])
            for p, previous in zip(model.parameters(), query, strict=True):
                torch.testing.assert_close(p, previous)
        with mock.patch("lab.infrastructure.engine.eager.evaluate", side_effect=RuntimeError("observe")):
            with self.assertRaisesRegex(RuntimeError, "observe"):
                stepper.evaluate("heldout")
        for p, previous in zip(model.parameters(), query, strict=True):
            torch.testing.assert_close(p, previous)

    def test_compiled_forward_with_clipping_follows_eager_updates(self):
        spec = modular(gptmini(8, 1, 2), prime=5, updates=4, batch=4, every=2)
        spec = swap(spec, "optimizer", Clipped(experiment().optimizer, norm=.25))
        task = build_task(spec.benchmark, 0, torch.device("cpu"))
        model = build_model(spec.model, task.vocab, 0)
        reference = build_model(spec.model, task.vocab, 0)
        compiled = CompiledStepper(swap(spec, "execution", Compiled()), task, model, None)
        eager = EagerStepper(spec, task, reference, None)
        parts = ((torch.arange(4), 0, 4),)
        for rate in (0., .005, .02):
            for stepper in (compiled, eager):
                stepper.step(parts, rate, False)
            for p, q in zip(model.parameters(), reference.parameters(), strict=True):
                torch.testing.assert_close(p, q, rtol=1e-4, atol=1e-6)
        for split in ("train", "heldout"):
            self.assertAlmostEqual(compiled.evaluate(split)["loss"], eager.evaluate(split)["loss"], places=5)
        self.assertTrue(compiled.optimizer.param_groups[0]["training"])


if __name__ == "__main__":
    unittest.main()
