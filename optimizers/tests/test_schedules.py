"""Analytic schedule values, horizons and optimizer-independent compatibility."""

from __future__ import annotations

import math
import unittest

import torch
from _warmup_cases import gradients, parameters

import lean_optim as opt


class ScheduleTests(unittest.TestCase):
    """Check warmup boundaries, independent parameter groups and decayed endpoints."""

    def test_rates_match_analytic_values_in_every_parameter_group(self) -> None:
        """Check all transition points and the clamped tail against explicit values."""
        expected = {
            "constant": (0, 0.5, 1, 1, 1, 1, 1, 1),
            "linear": (0, 0.5, 1, 0.75, 0.5, 0.25, 0, 0),
            "cosine": (0, 0.5, 1, (2 + math.sqrt(2)) / 4, 0.5, (2 - math.sqrt(2)) / 4, 0, 0),
        }
        for decay, factors in expected.items():
            for floor in (0.0, 0.25, 1.0):
                with self.subTest(decay=decay, floor=floor):
                    values = parameters()
                    optimizer = opt.standard.SGD(
                        [
                            {"params": [values[0]], "lr": 0.125},
                            {"params": [values[1]], "lr": 0.0625},
                        ]
                    )
                    scheduler = opt.LearningRateSchedule(
                        optimizer, warmup=2, total_steps=6, decay=decay, final_factor=floor
                    )
                    for index, factor in enumerate(factors):
                        adjusted = (
                            floor + (1 - floor) * factor if index >= scheduler.warmup else factor
                        )
                        rates = scheduler.get_last_lr()
                        for actual, peak in zip(rates, (0.125, 0.0625), strict=True):
                            self.assertAlmostEqual(actual, peak * adjusted)
                        optimizer.step()
                        scheduler.step()

    def test_decay_without_warmup_and_zero_peak(self) -> None:
        """Start at the full rate without warmup and handle an identically zero rate."""
        for peak in (0.0, 0.125):
            optimizer = opt.Adam(parameters(), lr=peak)
            scheduler = opt.LearningRateSchedule(optimizer, total_steps=4, decay="linear")
            for factor in (1.0, 0.75, 0.5, 0.25, 0.0, 0.0):
                self.assertEqual(scheduler.get_last_lr(), [peak * factor])
                gradients(optimizer, 0.125)
                optimizer.step()
                scheduler.step()

    def test_builtin_warmup_allows_a_decay_schedule_without_outer_warmup(self) -> None:
        """Built-in Schedule-Free warmup can be combined with an explicit outer decay."""
        for factory in (
            opt.ScheduleFreeAdamW,
            opt.ScheduleFreeAMSGradW,
            opt.ScheduleFreeAMSGradMD,
        ):
            with self.subTest(rule=factory.__name__):
                optimizer = factory(parameters(), lr=0.125, warmup=3)
                scheduler = opt.LearningRateSchedule(optimizer, total_steps=8, decay="linear")
                for _ in range(10):
                    gradients(optimizer, 0.125)
                    optimizer.step()
                    scheduler.step()
                self.assertEqual(scheduler.get_last_lr(), [0.0])
                self.assertTrue(
                    all(bool(p.isfinite().all()) for p in optimizer.param_groups[0]["params"])
                )

    def test_native_checkpoint_accepts_an_empty_parameter_group(self) -> None:
        """Restore tensor rates even when a native group has no reference parameter."""
        original = opt.standard.SGD(
            [
                {"params": parameters(), "lr": torch.tensor(0.125)},
                {"params": [], "lr": torch.tensor(0.0625)},
            ]
        )
        schedule = opt.LearningRateSchedule(original, total_steps=4, decay="linear")
        original.step()
        schedule.step()
        restored = opt.standard.SGD([{"params": parameters()}, {"params": []}], lr=0.5)
        continued = opt.LearningRateSchedule(restored)
        restored.load_state_dict(original.state_dict())
        continued.load_state_dict(schedule.state_dict())
        self.assertEqual(continued.get_last_lr(), schedule.get_last_lr())
        continued.step()
        self.assertEqual(continued.get_last_lr(), [0.0625, 0.03125])

    def test_invalid_schedules_leave_optimizer_rates_untouched(self) -> None:
        """Reject invalid horizons, decay names and rates before the initial zero step."""
        cases = (
            {"decay": "exponential"},
            {"decay": "linear"},
            {"decay": "cosine"},
            {"total_steps": 0},
            {"total_steps": True},
            {"total_steps": 4.5},
            {"warmup": 4, "total_steps": 4},
            {"warmup": 4, "total_steps": 3},
            {"final_factor": -0.5},
            {"final_factor": 1.5},
            {"final_factor": math.nan},
        )
        for options in cases:
            optimizer = opt.SGD(parameters(), lr=0.125)
            with self.subTest(options=options), self.assertRaises(ValueError):
                opt.LearningRateSchedule(optimizer, **options)
            self.assertEqual(optimizer.param_groups[0]["lr"], 0.125)
            self.assertNotIn("initial_lr", optimizer.param_groups[0])
        for rate in (-0.125, math.inf, math.nan):
            optimizer = opt.SGD(parameters())
            optimizer.param_groups[0]["lr"] = rate
            with self.subTest(rate=rate), self.assertRaises(ValueError):
                opt.LearningRateSchedule(optimizer, warmup=4)


if __name__ == "__main__":
    unittest.main()
