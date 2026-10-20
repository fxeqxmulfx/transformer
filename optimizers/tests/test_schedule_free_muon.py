"""Composition tests against native Muon and explicit complete-trajectory means."""

from __future__ import annotations

import copy
import math
import unittest

import torch

from lean_optim import Muon, ScheduleFreeMuon


def parameters(device: str = "cpu") -> list[torch.Tensor]:
    """Include tall/wide selected matrices, an excluded matrix and a vector."""
    return [
        torch.nn.Parameter(
            torch.linspace(-0.4, 0.7, math.prod(shape), device=device, dtype=torch.float64).reshape(
                shape
            )
        )
        for shape in ((2, 3), (3, 2), (2, 2), (2,))
    ]


class ScheduleFreeMuonTests(unittest.TestCase):
    """Verify oracle locations, retained inner histories and serialized modes."""

    def test_trajectory_is_native_muon_at_query_gradients_with_full_prefix_means(self) -> None:
        """Native Muon is independently covered by bundled historical references."""
        devices = ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",)
        for device in devices:
            for beta in (0.0, 0.6, 1.0):
                for warmup in (0, 3):
                    with self.subTest(device=device, beta=beta, warmup=warmup):
                        actual, bases = parameters(device), parameters(device)
                        optimizer = ScheduleFreeMuon(
                            actual,
                            lr=0.2,
                            beta=beta,
                            momentum=0.75,
                            auxiliary=0.3,
                            weight_decay=0.4,
                            warmup=warmup,
                            matrix_params=actual[:2],
                        )
                        native = Muon(
                            bases, lr=0.2, momentum=0.75, auxiliary=0.3, matrix_params=bases[:2]
                        )
                        history = [[p.detach().clone()] for p in bases]
                        for t, rate in enumerate((0.0, 0.2, 0.1, 0.0, 0.15), 1):
                            for p, z, prefix in zip(actual, bases, history, strict=True):
                                mean = torch.stack(prefix).mean(0)
                                torch.testing.assert_close(p, beta * mean + (1 - beta) * z)
                            optimizer.zero_grad(set_to_none=True)
                            sum(p.square().sum() / 2 for p in actual).backward()
                            for p, z in zip(actual, bases, strict=True):
                                z.grad = p.grad.detach().clone()
                            eta = rate * (min(1, t / warmup) if warmup else 1)
                            with torch.no_grad():
                                for z in bases:
                                    z.mul_(1 - eta * 0.4)
                            native.param_groups[0]["lr"] = eta
                            native.step()
                            optimizer.param_groups[0]["lr"] = rate
                            optimizer.step()
                            for p, z, prefix in zip(actual, bases, history, strict=True):
                                prefix.append(z.detach().clone())
                                mean = torch.stack(prefix).mean(0)
                                torch.testing.assert_close(optimizer.state[p]["base"], z)
                                torch.testing.assert_close(optimizer.state[p]["evaluation"], mean)
                                torch.testing.assert_close(p, beta * mean + (1 - beta) * z)
                            optimizer.eval()
                            for p, prefix in zip(actual, history, strict=True):
                                torch.testing.assert_close(p, torch.stack(prefix).mean(0))
                            optimizer.train()
                        self.assertIn("momentum", optimizer.state[actual[0]])
                        self.assertIn("m", optimizer.state[actual[2]])
                        self.assertNotIn("momentum", optimizer.state[actual[2]])

    def test_frozen_lean_decay_case_includes_initial_point_and_uses_base_decay(self) -> None:
        """Check exact rational values from muonWrapper_first_decay_fixture."""
        p = torch.nn.Parameter(torch.ones((1, 1), dtype=torch.float64))
        optimizer = ScheduleFreeMuon([p], lr=0.25, beta=0.5, momentum=0, weight_decay=0.5)
        p.grad = torch.zeros_like(p)
        optimizer.step()
        self.assertEqual(optimizer.state[p]["base"].item(), 7 / 8)
        self.assertEqual(optimizer.state[p]["evaluation"].item(), 15 / 16)
        self.assertEqual(p.item(), 29 / 32)
        optimizer.step()
        self.assertAlmostEqual(optimizer.state[p]["base"].item(), 49 / 64)
        self.assertAlmostEqual(optimizer.state[p]["evaluation"].item(), 169 / 192)
        self.assertAlmostEqual(p.item(), 79 / 96)

    def test_checkpoints_restore_modes_selection_clocks_and_all_hyperparameters(self) -> None:
        """Resume even when constructor matrix selection and hyperparameters differ."""
        for evaluation in (False, True):
            with self.subTest(evaluation=evaluation):
                original = parameters()
                optimizer = ScheduleFreeMuon(
                    original,
                    lr=0.03,
                    beta=0.4,
                    momentum=0.7,
                    auxiliary=0.2,
                    weight_decay=0.5,
                    warmup=4,
                    matrix_params=original[:2],
                )
                for _ in range(3):
                    for p in original:
                        p.grad = p.detach().sin()
                    optimizer.step()
                if evaluation:
                    optimizer.eval()
                restored = [torch.nn.Parameter(p.detach().clone()) for p in original]
                resumed = ScheduleFreeMuon(restored, lr=0.9, beta=1, matrix_params=[])
                resumed.load_state_dict(copy.deepcopy(optimizer.state_dict()))
                self.assertEqual(resumed.param_groups[0]["training"], not evaluation)
                for _ in range(3):
                    for rule, values in ((optimizer, original), (resumed, restored)):
                        rule.train()
                        for p in values:
                            p.grad = p.detach().sin()
                        rule.step()
                    for p, q in zip(original, restored, strict=True):
                        self.assertTrue(torch.equal(p, q))
                self.assertEqual(resumed.state[restored[2]]["step"].dtype, torch.float64)
                self.assertTrue(resumed.state[restored[0]]["matrix_update"])
                self.assertFalse(resumed.state[restored[2]]["matrix_update"])

    def test_closure_and_mode_errors_preserve_the_clock_and_queries(self) -> None:
        """Validate every gradient before advancing the common wrapper clock."""
        values = parameters()
        optimizer = ScheduleFreeMuon(values)
        values[0].grad = torch.ones_like(values[0])
        with self.assertRaisesRegex(ValueError, "dense gradient"):
            optimizer.step()
        self.assertEqual(optimizer.param_groups[0]["updates"], 0)

        def closure() -> torch.Tensor:
            optimizer.zero_grad(set_to_none=True)
            loss = sum(p.square().sum() for p in values)
            loss.backward()
            return loss

        result = optimizer.step(closure)
        self.assertIsNotNone(result)
        self.assertEqual(optimizer.param_groups[0]["updates"], 1)
        optimizer.eval()
        before = [p.detach().clone() for p in values]
        with self.assertRaisesRegex(RuntimeError, "train"):
            optimizer.step(closure)
        for p, previous in zip(values, before, strict=True):
            self.assertTrue(torch.equal(p, previous))
        self.assertEqual(optimizer.param_groups[0]["updates"], 1)

    def test_invalid_options_and_sparse_gradients_fail_before_updates(self) -> None:
        """Reject invalid scalar settings, matrix membership and missing dense data."""
        p = torch.nn.Parameter(torch.ones((2, 2)))
        for options in (
            {"beta": math.nan},
            {"beta": -0.1},
            {"beta": 1.1},
            {"lr": math.inf},
            {"momentum": 1.0},
            {"momentum": math.nan},
            {"auxiliary": 0},
            {"weight_decay": -1},
            {"weight_decay": math.inf},
            {"warmup": -1},
            {"warmup": True},
            {"matrix_params": [torch.ones(2)]},
            {"matrix_params": [p, p]},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                ScheduleFreeMuon([p], **options)
        optimizer = ScheduleFreeMuon([p])
        p.grad = torch.ones_like(p).to_sparse()
        with self.assertRaisesRegex(ValueError, "dense gradient"):
            optimizer.step()
        self.assertEqual(optimizer.param_groups[0]["updates"], 0)


if __name__ == "__main__":
    unittest.main()
