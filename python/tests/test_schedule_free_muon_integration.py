"""Schedule-Free Muon selection, execution, evaluation and lab checkpoints."""

import copy
import math
import unittest
from unittest import mock

import lean_optim
import torch

from lab.dsl import (Clipped, Compiled, CudaGraph, Guarded, Magma, ScheduleFreeMuon,
                     describe, swap)
from lab.domain.optimizers import Optimizer
from lab.domain.spec import blocks_of
from lab.infrastructure.benchmarks import build_task
from lab.infrastructure.engine.compiled import CompiledStepper
from lab.infrastructure.engine.eager import EagerStepper
from lab.infrastructure.nn import build_model
from lab.infrastructure.optim import build_optimizer

from examples import gptmini, modular
from test_schedule_free_integration import experiment as adam_experiment


def experiment(warmup=0):
    return swap(adam_experiment(warmup), "optimizer",
                ScheduleFreeMuon(lr=.02, beta=.6, momentum=.75, auxiliary=.3,
                                 weight_decay=.3, warmup=warmup))


class ScheduleFreeMuonIntegrationTests(unittest.TestCase):
    def test_dsl_selection_and_invalid_execution_fail_before_training(self):
        base = experiment()
        self.assertIn(ScheduleFreeMuon, blocks_of(Optimizer))
        self.assertEqual(describe(base.optimizer)["type"], "ScheduleFreeMuon")
        for options in ({"lr": 0}, {"lr": math.inf}, {"beta": math.nan}, {"beta": 1.1},
                        {"momentum": 1}, {"auxiliary": 0}, {"weight_decay": -1},
                        {"warmup": -1}, {"warmup": True}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                ScheduleFreeMuon(**({"lr": .01} | options))
        for stage in (Guarded, Magma):
            with self.subTest(stage=stage.__name__), self.assertRaisesRegex(ValueError, "Schedule-Free"):
                stage(base.optimizer)
        with self.assertRaisesRegex(ValueError, "Schedule-Free"):
            swap(base, "execution", CudaGraph())
        with self.assertRaisesRegex(ValueError, "warmup"):
            swap(base, "optimizer.warmup", 3)
        spec = swap(modular(gptmini(8, 1, 2), prime=5, updates=4, batch=4, every=2),
                    "optimizer", base.optimizer)
        model = build_model(spec.model, vocab=5, seed=0)
        rule = build_optimizer(base.optimizer, model)
        selected = {p for p in model.parameters() if rule.state[p]["matrix_update"]}
        self.assertEqual(selected, set(model.hidden_matrices()))
        self.assertTrue(any(p.ndim == 2 and p not in selected for p in model.parameters()))
        with self.assertRaisesRegex(ValueError, "eager"):
            build_optimizer(base.optimizer, model, rate=torch.tensor(0.))

    def test_eager_schedules_builtin_warmup_and_eval_checkpoint_resume(self):
        for warmup in (0, 3):
            with self.subTest(warmup=warmup):
                spec = experiment(warmup)
                task = build_task(spec.benchmark, 0, torch.device("cpu"))
                model, reference = (build_model(spec.model, task.vocab, 0) for _ in range(2))
                stepper = EagerStepper(spec, task, model, None)
                manual = lean_optim.ScheduleFreeMuon(
                    reference, **vars(spec.optimizer))
                parts = ((torch.arange(4), 0, 4),)
                for rate in (0., .005, .02, 0., .01):
                    manual.param_groups[0]["lr"] = rate
                    manual.zero_grad(set_to_none=True)
                    task.loss(*task.forward(reference, task.inputs(torch.arange(4)))).backward()
                    manual.step()
                    stepper.step(parts, rate, False)
                    for p, q in zip(model.parameters(), reference.parameters(), strict=True):
                        self.assertTrue(torch.equal(p, q))
                stepper.optimizer.eval()
                checkpoint = copy.deepcopy(stepper.state_dict())
                network = build_model(spec.model, task.vocab, 0)
                network.load_state_dict(copy.deepcopy(model.state_dict()))
                resumed = EagerStepper(spec, task, network, None)
                resumed.load_state_dict(checkpoint)
                self.assertFalse(resumed.optimizer.param_groups[0]["training"])
                for solver in (stepper, resumed):
                    solver.step(parts, .015, False)
                for p, q in zip(model.parameters(), network.parameters(), strict=True):
                    self.assertTrue(torch.equal(p, q))

    def test_evaluations_install_means_and_restore_queries_even_after_failure(self):
        spec = experiment()
        task = build_task(spec.benchmark, 0, torch.device("cpu"))
        model = build_model(spec.model, task.vocab, 0)
        stepper = EagerStepper(spec, task, model, None)
        for rate in (.01, .02, .005):
            stepper.step(((torch.arange(4), 0, 4),), rate, False)
        query = [p.detach().clone() for p in model.parameters()]

        def observe(task, network, rows, batch):
            self.assertFalse(stepper.optimizer.param_groups[0]["training"])
            for p in network.parameters():
                self.assertTrue(torch.equal(p, stepper.optimizer.state[p]["evaluation"]))
            return {"observations": len(rows)}

        for split in ("train", "heldout"):
            with mock.patch("lab.infrastructure.engine.eager.evaluate", side_effect=observe):
                self.assertGreater(stepper.evaluate(split)["observations"], 0)
        with mock.patch("lab.infrastructure.engine.eager.evaluate", side_effect=RuntimeError("observe")):
            with self.assertRaisesRegex(RuntimeError, "observe"):
                stepper.evaluate("heldout")
        self.assertTrue(stepper.optimizer.param_groups[0]["training"])
        for p, previous in zip(model.parameters(), query, strict=True):
            self.assertTrue(torch.equal(p, previous))

    def test_compiled_forward_with_clipping_matches_eager(self):
        spec = modular(gptmini(8, 1, 2), prime=5, updates=4, batch=4, every=2)
        spec = swap(spec, "optimizer", Clipped(experiment().optimizer, norm=.25))
        task = build_task(spec.benchmark, 0, torch.device("cpu"))
        model, reference = (build_model(spec.model, task.vocab, 0) for _ in range(2))
        compiled = CompiledStepper(swap(spec, "execution", Compiled()), task, model, None)
        eager = EagerStepper(spec, task, reference, None)
        for rate in (0., .005, .02):
            for stepper in (compiled, eager):
                stepper.step(((torch.arange(4), 0, 4),), rate, False)
            for p, q in zip(model.parameters(), reference.parameters(), strict=True):
                torch.testing.assert_close(p, q, rtol=1e-4, atol=1e-6)
        for split in ("train", "heldout"):
            self.assertAlmostEqual(compiled.evaluate(split)["loss"], eager.evaluate(split)["loss"], places=5)


if __name__ == "__main__":
    unittest.main()
