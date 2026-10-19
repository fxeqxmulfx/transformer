"""MDW checkpoint continuation, mode persistence and kernel-checked literal values."""

from __future__ import annotations

import copy
import io
import math
import unittest
from itertools import product

import torch
from _mdw_cases import NAMES, Run

import lean_optim as opt


class MDWStateTests(unittest.TestCase):
    """Keep saved rates, histories, factor selection and physical states consistent."""

    def test_checkpoint_continues_in_training_and_evaluation_modes(self) -> None:
        """Round-trip pure state through torch.save, including live destination rate storage."""
        for name, evaluation in product(NAMES, (False, True)):
            with self.subTest(rule=name, evaluation=evaluation):
                original = Run.create(name)
                for _ in range(11):
                    original.step()
                free = name.startswith("ScheduleFree")
                if free and evaluation:
                    original.optimizer.eval()
                stream = io.BytesIO()
                torch.save(
                    (original.optimizer.state_dict(), original.scheduler.state_dict()), stream
                )
                stream.seek(0)
                state, schedule = torch.load(stream, weights_only=True)
                restored = Run.create(name)
                with torch.no_grad():
                    for p, q in zip(original.values, restored.values, strict=True):
                        q.copy_(p)
                restored.optimizer.load_state_dict(state)
                restored.scheduler.load_state_dict(schedule)
                self.assertFalse(restored.optimizer.workspace.buffers)
                if not free:
                    self.assertIs(restored.optimizer.param_groups[0]["lr"], restored.rate)
                if free:
                    self.assertEqual(restored.optimizer.param_groups[0]["training"], not evaluation)
                    original.optimizer.train()
                    restored.optimizer.train()
                for index in range(12):
                    for run in (original, restored):
                        run.gradients((-1) ** index / 8)
                        run.step()
                    self._assert_equal(original, restored)
                if free:
                    original.optimizer.eval()
                    restored.optimizer.eval()
                    self._assert_equal(original, restored)

    def test_wrong_variance_history_and_invalid_decay_are_rejected(self) -> None:
        """Reject incompatible algorithms before altering the destination's mathematical state."""
        for name in NAMES:
            for decay in (-0.1, math.inf, math.nan):
                with self.subTest(rule=name, decay=decay), self.assertRaises(ValueError):
                    getattr(opt, name)([torch.nn.Parameter(torch.ones(2, 3))], weight_decay=decay)
            other = (
                name.replace("AMSGrad", "Adam")
                if "AMSGrad" in name
                else name.replace("Adam", "AMSGrad")
            )
            original, incompatible = Run.create(name), Run.create(other)
            before = copy.deepcopy(incompatible.optimizer.state_dict())
            with self.assertRaisesRegex(ValueError, "same"):
                incompatible.optimizer.load_state_dict(original.optimizer.state_dict())
            self.assertEqual(
                incompatible.optimizer.state_dict()["param_groups"], before["param_groups"]
            )
            mode_name = (
                name.removeprefix("ScheduleFree")
                if name.startswith("ScheduleFree")
                else "ScheduleFree" + name
            )
            wrong_mode = Run.create(mode_name)
            with self.assertRaisesRegex(ValueError, "averaging mode"):
                wrong_mode.optimizer.load_state_dict(original.optimizer.state_dict())
            selected = getattr(opt, name)(original.values, matrix_params=[])
            with self.assertRaisesRegex(ValueError, "matrix selection"):
                selected.load_state_dict(original.optimizer.state_dict())

    def test_bias_corrections_and_physical_decay_match_frozen_lean_values(self) -> None:
        """The native and raw auxiliary values distinguish the two ordinary callbacks."""
        for name, expected in (("AdamMDW", 47 / 25), ("AMSGradMDW", 142 / 75)):
            p = torch.nn.Parameter(torch.tensor([2.0], dtype=torch.float64))
            optimizer = getattr(opt, name)(
                [p], lr=0.1, auxiliary_rate=0.1, betas=(0.5, 0.75), eps=0.5, weight_decay=0.2
            )
            p.grad = p.new_tensor([2.0])
            optimizer.step()
            self.assertAlmostEqual(float(p.detach()), expected, places=14)
        for name, expected in (("ScheduleFreeAdamMDW", 3 / 2), ("ScheduleFreeAMSGradMDW", 5 / 3)):
            p = torch.nn.Parameter(torch.tensor([2.0], dtype=torch.float64))
            optimizer = getattr(opt, name)(
                [p], lr=1.0, auxiliary_rate=1.0, betas=(0.5, 0.0), eps=1.0, weight_decay=0.0
            )
            state = optimizer.state[p]
            state.update(second=p.new_tensor([4.0]), total=1.0, step=1)
            if "maximum" in state:
                state["maximum"].fill_(4.0)
            p.grad = p.new_tensor([1.0])
            optimizer.step()
            self.assertAlmostEqual(float(state["base"]), expected, places=14)

    def test_zero_gradient_still_decays_the_physical_weight(self) -> None:
        """Lean's physical MDW checkpoint is 2 -> 3/2 with no gradient contribution."""
        for name in NAMES:
            p = torch.nn.Parameter(torch.tensor([[2.0]], dtype=torch.float64))
            optimizer = getattr(opt, name)(
                [p], lr=0.25, gain_rate=0.125, betas=(0.5, 0.0), eps=1.0, weight_decay=1.0
            )
            p.grad = torch.zeros_like(p)
            optimizer.step()
            torch.testing.assert_close(p, p.new_tensor([[1.5]]), rtol=0, atol=1e-14)
            if name.startswith("ScheduleFree"):
                optimizer.eval()
                torch.testing.assert_close(p, p.new_tensor([[1.5]]), rtol=0, atol=1e-14)

    def _assert_equal(self, original: Run, restored: Run) -> None:
        for p, q in zip(original.values, restored.values, strict=True):
            torch.testing.assert_close(p, q, rtol=0, atol=0)
