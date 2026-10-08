"""Scientific controls for same-gradient, isolated native-moment resets."""

import copy
import hashlib
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
from moment_resets import BRANCHES, measure, reset_state, restored_optimizer
from momentum_directions import losses


class MomentResetTests(unittest.TestCase):
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

    def equal(self, left, right):
        if isinstance(left, torch.Tensor):
            self.assertTrue(torch.equal(left, right))
        elif isinstance(left, dict):
            self.assertEqual(set(left), set(right))
            for key in left:
                self.equal(left[key], right[key])
        elif isinstance(left, (tuple, list)):
            self.assertEqual(len(left), len(right))
            for first, second in zip(left, right):
                self.equal(first, second)
        else:
            self.assertEqual(left, right)

    def test_partial_resets_preserve_other_buffer_and_clock(self):
        for branch in BRANCHES:
            clone = copy.deepcopy(self.model)
            optimizer = restored_optimizer(self.spec, clone, self.snapshot)
            before = copy.deepcopy(optimizer.state_dict())
            reset_state(optimizer, branch)
            after = optimizer.state_dict()
            self.equal(before["param_groups"], after["param_groups"])
            for key, state in after["state"].items():
                old = before["state"][key]
                for buffer in ("exp_avg", "exp_avg_sq"):
                    reset = branch == "fresh_adamw" or branch == {"exp_avg": "reset_first_moment",
                        "exp_avg_sq": "reset_second_moment"}[buffer]
                    self.equal(state[buffer], torch.zeros_like(old[buffer]) if reset else old[buffer])
                self.equal(state["step"], torch.zeros_like(old["step"]) if branch == "fresh_adamw" else old["step"])

    def test_retained_and_fresh_branches_match_independent_native_updates(self):
        result = measure(self.model, self.task, self.spec, self.snapshot, batch=3)
        for branch in ("retained", "fresh_adamw"):
            clone = copy.deepcopy(self.model)
            eager = EagerStepper(self.spec, self.task, clone, None)
            if branch == "retained":
                eager.load_state_dict(copy.deepcopy(self.snapshot["optimizer"]))
            sampler = self.task.sampler(self.spec.budget.batch, self.spec.seeds.batch_seed)
            sampler.restore(copy.deepcopy(self.snapshot))
            parts, place = sampler.next()
            indices = gather(parts)
            digest = hashlib.sha256(indices.numpy().astype("<i8").tobytes()).hexdigest()
            self.assertEqual(result["minibatch"]["indices_sha256"], digest)
            self.assertEqual(result["minibatch"]["place"], place)
            eager.step(parts, rate(self.spec.optimizer.lr, self.spec.schedule, 2), False)
            for split in ("train_all", "heldout_nonzero"):
                rows = self.task.splits["train" if split == "train_all" else "heldout"]
                if split == "heldout_nonzero":
                    rows = rows[rows[:, 5] != self.task.corpus.tokens.index("0")]
                expected = losses(clone, self.task, rows, 3)
                self.equal(expected, result["branches"][branch]["populations"][split]["after"])
            self.assertEqual(result["branches"][branch]["clock_after"], 3 if branch == "retained" else 1)

    def test_preserves_supplied_state_gradients_modes_rng_and_next_update(self):
        self.model.train()
        self.model.blocks[0].eval()
        modes = [module.training for module in self.model.modules()]
        state = copy.deepcopy(self.model.state_dict())
        snapshot = copy.deepcopy(self.snapshot)
        optimizer = copy.deepcopy(self.stepper.state_dict())
        for parameter in self.model.parameters():
            parameter.grad = torch.ones_like(parameter)
        rng = torch.random.get_rng_state().clone()
        control = copy.deepcopy(self.model)
        measure(self.model, self.task, self.spec, self.snapshot, batch=3)
        self.equal(state, self.model.state_dict())
        self.equal(snapshot, self.snapshot)
        self.equal(optimizer, self.stepper.state_dict())
        self.assertEqual(modes, [module.training for module in self.model.modules()])
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        self.assertTrue(all(torch.equal(p.grad, torch.ones_like(p)) for p in self.model.parameters()))
        parts, _ = self.sampler.next()
        stepper = EagerStepper(self.spec, self.task, control, None)
        stepper.load_state_dict(copy.deepcopy(optimizer))
        self.stepper.step(parts, .001, False)
        stepper.step(parts, .001, False)
        self.equal(control.state_dict(), self.model.state_dict())
        self.equal(stepper.state_dict(), self.stepper.state_dict())

    def test_failure_restores_rng_modes_and_snapshot(self):
        modes = [module.training for module in self.model.modules()]
        snapshot = copy.deepcopy(self.snapshot)
        rng = torch.random.get_rng_state().clone()

        def fail(*args):
            torch.rand(1)
            raise RuntimeError("reset observation failed")

        with patch.object(self.task, "forward", fail), self.assertRaisesRegex(RuntimeError, "observation failed"):
            measure(self.model, self.task, self.spec, self.snapshot)
        self.equal(snapshot, self.snapshot)
        self.assertEqual(modes, [module.training for module in self.model.modules()])
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))

    def test_unarchived_clock_and_misregistration_are_rejected(self):
        snapshot = copy.deepcopy(self.snapshot)
        snapshot["step"] = 0
        with self.assertRaisesRegex(ValueError, "noninitial"):
            measure(self.model, self.task, self.spec, snapshot)
        snapshot = copy.deepcopy(self.snapshot)
        snapshot["optimizer"]["param_groups"][0]["params"].reverse()
        with self.assertRaisesRegex(ValueError, "registration"):
            measure(self.model, self.task, self.spec, snapshot)

    def test_first_moment_reset_changes_reachable_quadratic_ascent_to_descent(self):
        parameter = torch.nn.Parameter(torch.tensor([.00075], dtype=torch.float64))
        optimizer = torch.optim.AdamW([parameter], lr=.001, betas=(.9, .98), eps=1e-8, weight_decay=.1)
        (parameter.square().sum() / 2).backward()
        optimizer.step()
        baseline = parameter.detach().clone()
        saved = copy.deepcopy(optimizer.state_dict())
        old_loss = float(baseline.square().sum() / 2)
        for branch in ("retained", "reset_first_moment"):
            clone = torch.nn.Parameter(baseline.clone())
            observed = torch.optim.AdamW([clone], lr=.001, betas=(.9, .98), eps=1e-8, weight_decay=.1)
            observed.load_state_dict(copy.deepcopy(saved))
            reset_state(observed, branch)
            (clone.square().sum() / 2).backward()
            observed.step()
            new_loss = float(clone.detach().square().sum() / 2)
            self.assertGreater(new_loss, old_loss) if branch == "retained" else self.assertLess(new_loss, old_loss)

    def test_nonphysical_variance_is_rejected_before_any_reset(self):
        snapshot = copy.deepcopy(self.snapshot)
        state = next(iter(snapshot["optimizer"]["state"].values()))
        state["exp_avg_sq"].fill_(-1)
        with self.assertRaisesRegex(FloatingPointError, "moment"):
            measure(self.model, self.task, self.spec, snapshot)


if __name__ == "__main__":
    unittest.main()
