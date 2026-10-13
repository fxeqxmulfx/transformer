"""Native-state, exhaustive-gradient and restored-minibatch scientific controls."""

import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch

from lab.domain.spec import swap
from lab.domain.training import rate
from lab.infrastructure.benchmarks.modular import ModularTask
from lab.infrastructure.benchmarks.samplers import gather
from lab.infrastructure.engine.eager import EagerStepper
from lab.infrastructure.loader import load
from lab.infrastructure.nn import build_model

STUDY = Path(__file__).resolve().parents[2] / "experiments/grokking_internals"
sys.path.insert(0, str(STUDY))
from momentum_directions import gradients, measure, retained_direction


class MomentumDirectionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        spec = dict(load(STUDY).select([]))["gptmini-seed1"]
        for path, value in (("model.width", 16), ("model.block.attention.heads", 2),
                            ("benchmark.prime", 5), ("budget.batch", 3), ("schedule.warmup", 0)):
            spec = swap(spec, path, value)
        self.spec = spec
        self.task = ModularTask(spec.benchmark, spec.seeds.data, torch.device("cpu"))
        self.model = build_model(spec.model, self.task.vocab, spec.seeds.model).double()
        self.sampler = self.task.sampler(spec.budget.batch, spec.seeds.batch_seed)
        self.stepper = EagerStepper(spec, self.task, self.model, None)
        for step in range(2):
            parts, _ = self.sampler.next()
            self.stepper.step(parts, rate(spec.optimizer.lr, spec.schedule, step), False)
        self.snapshot = copy.deepcopy({"step": 2, "model": self.model.state_dict(),
                                       "optimizer": self.stepper.state_dict(), **self.sampler.state()})

    def equal(self, expected, actual):
        if isinstance(expected, torch.Tensor):
            self.assertTrue(torch.equal(expected, actual))
        elif isinstance(expected, dict):
            self.assertEqual(set(expected), set(actual))
            for key in expected:
                self.equal(expected[key], actual[key])
        elif isinstance(expected, (list, tuple)):
            self.assertEqual(len(expected), len(actual))
            for left, right in zip(expected, actual):
                self.equal(left, right)
        else:
            self.assertEqual(expected, actual)

    def test_exhaustive_gradient_is_sample_weighted_and_matches_full_batch_oracle(self):
        rows = self.task.splits["train"]
        stats, observed = gradients(self.model, self.task, rows, 3)
        output, target = self.task.forward(self.model, rows)
        full = self.task.loss(output, target)
        oracle = torch.cat([g.flatten() for g in torch.autograd.grad(full, tuple(self.model.parameters()))])
        torch.testing.assert_close(observed["full"], oracle, rtol=1e-10, atol=1e-12)
        self.assertAlmostEqual(stats["losses"]["full"], float(full.detach()), places=12)
        torch.testing.assert_close(observed["full"], (observed["answer"] + observed["EOS"]) / 2,
                                   rtol=1e-10, atol=1e-12)

    def test_disposable_step_agrees_with_independent_eager_update_and_sampler(self):
        result = measure(self.model, self.task, self.spec, self.snapshot, batch=3)
        clone = copy.deepcopy(self.model)
        eager = EagerStepper(self.spec, self.task, clone, None)
        eager.load_state_dict(copy.deepcopy(self.snapshot["optimizer"]))
        sampler = self.task.sampler(self.spec.budget.batch, self.spec.seeds.batch_seed)
        sampler.restore(copy.deepcopy(self.snapshot))
        parts, place = sampler.next()
        indices = gather(parts)
        self.assertEqual(result["minibatch"]["examples"], len(indices))
        self.assertEqual(result["minibatch"]["place"], place)
        eager.step(parts, rate(self.spec.optimizer.lr, self.spec.schedule, 2), False)
        clone.eval()
        for split in ("train", "heldout"):
            output, target = self.task.forward(clone, self.task.splits[split])
            actual = result["populations"][f"{split}_all"]["after"]
            self.assertAlmostEqual(actual["losses"]["full"], float(self.task.loss(output, target).detach()), places=12)
            self.assertEqual(actual["answer_accuracy"], float((output[:, 0].argmax(-1) == target[:, 0]).double().mean()))
        self.assertLess(result["finite_displacement_algorithm_relative_error"], 1e-10)

    def test_zero_quotient_is_removed_only_in_named_scopes(self):
        result = measure(self.model, self.task, self.spec, self.snapshot, batch=3)
        zero = self.task.corpus.tokens.index("0")
        for split, rows in self.task.splits.items():
            self.assertEqual(result["populations"][f"{split}_all"]["before"]["examples"], len(rows))
            self.assertEqual(result["populations"][f"{split}_nonzero"]["before"]["examples"], int((rows[:, 5] != zero).sum()))

    def test_preserves_model_snapshot_existing_gradients_modes_and_global_rng(self):
        self.model.train()
        self.model.blocks[0].eval()
        modes = [module.training for module in self.model.modules()]
        state = copy.deepcopy(self.model.state_dict())
        snapshot = copy.deepcopy(self.snapshot)
        for parameter in self.model.parameters():
            parameter.grad = torch.ones_like(parameter)
        rng = torch.random.get_rng_state().clone()
        measure(self.model, self.task, self.spec, self.snapshot, batch=3)
        self.equal(state, self.model.state_dict())
        self.equal(snapshot, self.snapshot)
        self.assertEqual(modes, [module.training for module in self.model.modules()])
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        self.assertTrue(all(torch.equal(p.grad, torch.ones_like(p)) for p in self.model.parameters()))

    def test_failure_restores_modes_rng_and_archived_state(self):
        modes = [module.training for module in self.model.modules()]
        rng = torch.random.get_rng_state().clone()
        snapshot = copy.deepcopy(self.snapshot)

        def fail(*args):
            torch.rand(1)
            raise RuntimeError("moment probe failed")

        with patch.object(self.task, "forward", fail), self.assertRaisesRegex(RuntimeError, "moment probe failed"):
            measure(self.model, self.task, self.spec, self.snapshot)
        self.equal(snapshot, self.snapshot)
        self.assertEqual(modes, [module.training for module in self.model.modules()])
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))

    def test_initial_missing_moments_and_misregistered_parameters_are_rejected(self):
        snapshot = copy.deepcopy(self.snapshot)
        snapshot["step"] = 0
        with self.assertRaisesRegex(ValueError, "noninitial"):
            measure(self.model, self.task, self.spec, snapshot)
        snapshot = copy.deepcopy(self.snapshot)
        snapshot["optimizer"]["param_groups"][0]["params"].reverse()
        with self.assertRaisesRegex(ValueError, "registration"):
            measure(self.model, self.task, self.spec, snapshot)

    def test_nonphysical_variance_and_inconsistent_clocks_are_rejected(self):
        state = self.stepper.optimizer.state[next(self.model.parameters())]
        state["exp_avg_sq"].fill_(-1)
        with self.assertRaisesRegex(FloatingPointError, "moment"):
            retained_direction(self.stepper.optimizer, 2)
        with self.assertRaisesRegex(ValueError, "clock"):
            retained_direction(self.stepper.optimizer, 3)

    def test_reachable_scalar_quadratic_retains_opposed_momentum_and_ascends(self):
        parameter = torch.nn.Parameter(torch.tensor([.00075], dtype=torch.float64))
        optimizer = torch.optim.AdamW([parameter], lr=.001, betas=(.9, .98), eps=1e-8, weight_decay=.1)
        (parameter.square().sum() / 2).backward()
        optimizer.step()
        self.assertLess(float(parameter.detach()), 0)
        old, _ = retained_direction(optimizer, 1)
        self.assertGreater(float(old), 0)
        loss = float(parameter.detach().square().sum() / 2)
        optimizer.zero_grad(set_to_none=True)
        (parameter.square().sum() / 2).backward()
        current = parameter.grad.detach().clone()
        optimizer.step()
        new, _ = retained_direction(optimizer, 2)
        self.assertLess(float(current @ new), 0)
        self.assertGreater(float(parameter.detach().square().sum() / 2), loss)


if __name__ == "__main__":
    unittest.main()
