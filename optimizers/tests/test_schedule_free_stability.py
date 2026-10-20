"""Floating-point fixed points and the Schedule-Free workspace memory budget."""

from __future__ import annotations

import unittest

import torch

from lean_optim import (
    ScheduleFreeAdamW,
    ScheduleFreeAdamWPaper,
    ScheduleFreeAMSGradW,
    ScheduleFreeSGD,
)


class ScheduleFreeStabilityTests(unittest.TestCase):
    """Protect exact stationary iterates and bounded, reusable scratch storage."""

    def test_initial_train_eval_round_trips_preserve_float32_values(self) -> None:
        """Equal base and average must reproduce the initial query exactly."""
        for factory in (ScheduleFreeSGD, ScheduleFreeAdamW, ScheduleFreeAMSGradW):
            with self.subTest(rule=factory.__name__):
                initial = torch.tensor([1.5, -1.5, 0.7, -0.3], dtype=torch.float32)
                parameter = torch.nn.Parameter(initial.clone())
                optimizer = factory([parameter])
                for _ in range(4):
                    optimizer.eval()
                    self.assertTrue(torch.equal(parameter, initial))
                    optimizer.train()
                    self.assertTrue(torch.equal(parameter, initial))

    def test_stationary_base_is_preserved_by_weighted_averaging(self) -> None:
        """Zero gradients and decay leave equal iterates fixed at positive rates."""
        for factory in (ScheduleFreeSGD, ScheduleFreeAdamW, ScheduleFreeAMSGradW):
            with self.subTest(rule=factory.__name__):
                initial = torch.tensor([1.5, -1.5, 0.7, -0.3], dtype=torch.float32)
                parameter = torch.nn.Parameter(initial.clone())
                optimizer = factory([parameter], lr=0.1)
                parameter.grad = torch.zeros_like(parameter)
                for _ in range(20):
                    optimizer.step()
                    for key in ("base", "evaluation"):
                        if key in optimizer.state[parameter]:
                            self.assertTrue(torch.equal(optimizer.state[parameter][key], initial))
                    self.assertTrue(torch.equal(parameter, initial))

    def test_zero_rates_preserve_initial_and_later_queries_while_moments_advance(self) -> None:
        """Pause AdamW averaging at zero weight without stopping its variance clock."""
        for factory in (ScheduleFreeAdamWPaper, ScheduleFreeAMSGradW):
            with self.subTest(rule=factory.__name__):
                parameter = torch.nn.Parameter(torch.tensor([1.5, -0.7], dtype=torch.float32))
                optimizer = factory([parameter], lr=0.0, weight_decay=0.3)
                for rate in (0.0, 0.1, 0.2, 0.0, 0.0):
                    before = parameter.detach().clone()
                    state = optimizer.state[parameter]
                    base, evaluation = state["base"].clone(), state["evaluation"].clone()
                    total, step = state["total"], state["step"]
                    optimizer.param_groups[0]["lr"] = rate
                    parameter.grad = torch.full_like(parameter, step + 1.0)
                    second = state["second"].clone()
                    optimizer.step()
                    self.assertEqual(state["step"], step + 1)
                    self.assertFalse(torch.equal(state["second"], second))
                    if not rate:
                        self.assertTrue(torch.equal(parameter, before))
                        self.assertTrue(torch.equal(state["base"], base))
                        self.assertTrue(torch.equal(state["evaluation"], evaluation))
                        self.assertEqual(state["total"], total)

    def test_adamw_uses_at_most_one_parameter_sized_scratch_buffer(self) -> None:
        """Keep the query, two iterates and variance without a second work tensor."""
        for factory in (ScheduleFreeAdamW, ScheduleFreeAMSGradW):
            with self.subTest(rule=factory.__name__):
                parameter = torch.nn.Parameter(torch.full((16, 24), 0.25))
                parameter.grad = torch.full_like(parameter, 0.125)
                optimizer = factory([parameter], weight_decay=0.1)
                for _ in range(3):
                    optimizer.step()
                buffers = [
                    buffer
                    for group in optimizer.workspace.buffers.values()
                    for buffer in group.values()
                ]
                scratch_bytes = sum(buffer.numel() * buffer.element_size() for buffer in buffers)
                self.assertLessEqual(scratch_bytes, parameter.numel() * parameter.element_size())
                addresses = [buffer.data_ptr() for buffer in buffers]
                optimizer.eval()
                optimizer.train()
                optimizer.step()
                self.assertEqual(
                    addresses,
                    [
                        buffer.data_ptr()
                        for group in optimizer.workspace.buffers.values()
                        for buffer in group.values()
                    ],
                )


if __name__ == "__main__":
    unittest.main()
