"""MDW DSL blocks, real lab updates and Schedule-Free evaluation weights."""

import unittest
from unittest import mock

import lean_optim
import torch

from lab.dsl import (AMSGradMDW, AdamMDW, CudaGraph, Guarded, Magma, Schedule,
                     ScheduleFreeAMSGradMDW, ScheduleFreeAdamMDW, swap)
from lab.infrastructure.benchmarks import build_task
from lab.infrastructure.engine.eager import EagerStepper
from lab.infrastructure.nn import build_model

from examples import gptmini, modular


class MDWIntegrationTests(unittest.TestCase):
    def test_four_blocks_build_and_warmup_freezes_every_physical_tensor(self):
        for cls in (AMSGradMDW, AdamMDW, ScheduleFreeAMSGradMDW, ScheduleFreeAdamMDW):
            with self.subTest(rule=cls.__name__):
                experiment = modular(gptmini(8, 1, 2), prime=5, updates=4, batch=4, every=2)
                experiment = swap(experiment, "optimizer", cls(lr=.01, weight_decay=.25))
                task = build_task(experiment.benchmark, 0, torch.device("cpu"))
                model = build_model(experiment.model, task.vocab, 0)
                stepper = EagerStepper(experiment, task, model, None)
                self.assertIsInstance(stepper.optimizer, getattr(lean_optim, cls.__name__))
                self.assertEqual(stepper.optimizer.param_groups[0]["initial_lr"], .01)
                old = [p.detach().clone() for p in model.parameters()]
                parts = ((torch.arange(4), 0, 4),)
                stepper.step(parts, 0., False)
                self.assertTrue(all(torch.equal(p, before) for p, before in zip(model.parameters(), old)))
                for _ in range(3):
                    stepper.step(parts, .01, False)
                if isinstance(stepper.optimizer, lean_optim.ScheduleFreeSGD):
                    query = [p.detach().clone() for p in model.parameters()]

                    def check_evaluation(task, network, rows, batch):
                        self.assertFalse(stepper.optimizer.param_groups[0]["training"])
                        return {"observations": len(rows)}

                    with mock.patch("lab.infrastructure.engine.eager.evaluate", side_effect=check_evaluation):
                        result = stepper.evaluate("train")
                    self.assertGreater(result["observations"], 0)
                    self.assertTrue(stepper.optimizer.param_groups[0]["training"])
                    self.assertTrue(all(torch.equal(p, before) for p, before in zip(model.parameters(), query)))

    def test_schedule_free_modes_restore_even_after_an_evaluation_exception(self):
        from lab.infrastructure.optim import evaluation_weights, training_weights

        parameter = torch.nn.Parameter(torch.ones(2, 3))
        optimizer = lean_optim.ScheduleFreeAdamMDW([parameter])
        with self.assertRaisesRegex(RuntimeError, "observation failed"):
            with evaluation_weights(optimizer):
                self.assertFalse(optimizer.param_groups[0]["training"])
                raise RuntimeError("observation failed")
        self.assertTrue(optimizer.param_groups[0]["training"])
        optimizer.eval()
        training_weights(optimizer)
        self.assertTrue(optimizer.param_groups[0]["training"])

    def test_invalid_stage_and_warmup_combinations_fail_at_configuration_time(self):
        base = modular(gptmini(8, 1, 2), prime=5, updates=4, batch=4, every=2)
        for cls in (AMSGradMDW, AdamMDW, ScheduleFreeAMSGradMDW, ScheduleFreeAdamMDW):
            with self.subTest(rule=cls.__name__):
                spec = cls(lr=.01)
                with self.assertRaisesRegex(ValueError, "MDW"):
                    Guarded(spec)
                with self.assertRaisesRegex(ValueError, "MDW"):
                    Magma(spec)
        for cls in (ScheduleFreeAMSGradMDW, ScheduleFreeAdamMDW):
            experiment = swap(base, "optimizer", cls(lr=.01))
            with self.assertRaisesRegex(ValueError, "Schedule-Free MDW"):
                swap(experiment, "execution", CudaGraph())
            with self.assertRaisesRegex(ValueError, "warmup"):
                swap(base, "optimizer", cls(lr=.01, warmup=3))
            self.assertEqual(swap(swap(base, "schedule", Schedule()), "optimizer", cls(lr=.01, warmup=3)).optimizer.warmup, 3)
