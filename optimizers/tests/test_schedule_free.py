# Copyright (c) 2026 Mikhail Borisov
"""Actual query/evaluation buffers, indexing and Appendix D.1 projections."""

from __future__ import annotations

import copy
import math
import unittest

import torch

from lean_optim import ClippedOptimistic, ScheduleFreeAdamW, ScheduleFreeSGD


class ScheduleFreeTests(unittest.TestCase):
    """Numerical checks for schedulefree."""

    def test_sgd_average_includes_initial_point_and_beta_endpoints(self) -> None:
        """Check sgd average includes initial point and beta endpoints."""
        for beta in (0.0, 0.9, 1.0):
            p = torch.nn.Parameter(torch.tensor([2.0, -1.0], dtype=torch.float64))
            optimizer = ScheduleFreeSGD([p], lr=0.2, beta=beta)
            base, history = p.detach().clone(), [p.detach().clone()]
            for gradient in ([1.0, -2.0], [0.0, 3.0], [-1.0, 0.0]):
                p.grad = p.new_tensor(gradient)
                base = base - 0.2 * p.grad
                history.append(base.clone())
                expected = torch.stack(history).mean(0)
                optimizer.step()
                torch.testing.assert_close(p, beta * expected + (1 - beta) * base)
                optimizer.eval()
                torch.testing.assert_close(p, expected)
                optimizer.eval()
                torch.testing.assert_close(p, expected)
                with self.assertRaises(RuntimeError):
                    optimizer.step()
                optimizer.train()

    def test_adamw_uses_raw_variance_epsilon_query_decay_and_squared_effective_rate(self) -> None:
        """Check adamw uses raw variance epsilon query decay and squared effective rate."""
        for warmup in (0, 3):
            p = torch.nn.Parameter(torch.tensor([2.0, -1.0], dtype=torch.float64))
            optimizer = ScheduleFreeAdamW(
                [p], lr=0.2, betas=(0.6, 0.75), eps=0.125, weight_decay=0.3, warmup=warmup
            )
            base, evaluation, variance = [2.0, -1.0], [2.0, -1.0], [0.0, 0.0]
            total = 0.0
            for t, gradient in enumerate(([1.0, -2.0], [0.0, 3.0], [-1.0, 0.0]), 1):
                query = [0.6 * x + 0.4 * z for x, z in zip(evaluation, base, strict=False)]
                torch.testing.assert_close(p, p.new_tensor(query))
                eta = 0.2 * math.sqrt(1 - 0.75**t) * (min(1, t / warmup) if warmup else 1)
                total += eta**2
                c = eta**2 / total
                for i, g in enumerate(gradient):
                    variance[i] = 0.75 * variance[i] + 0.25 * g * g
                    base[i] -= eta * g / (math.sqrt(variance[i]) + 0.125) + eta * 0.3 * query[i]
                    evaluation[i] = (1 - c) * evaluation[i] + c * base[i]
                p.grad = p.new_tensor(gradient)
                optimizer.step()
                state = optimizer.state[p]
                torch.testing.assert_close(
                    state["base"], p.new_tensor(base), atol=1e-14, rtol=1e-14
                )
                torch.testing.assert_close(
                    state["evaluation"], p.new_tensor(evaluation), atol=1e-14, rtol=1e-14
                )
                self.assertAlmostEqual(state["total"], total)

    def test_eval_mode_checkpoint_and_zero_learning_rate(self) -> None:
        """Check eval mode checkpoint and zero learning rate."""
        p = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float64))
        optimizer = ScheduleFreeAdamW([p], lr=0.0)
        p.grad = torch.zeros_like(p)
        optimizer.step()
        self.assertTrue(bool(p.isfinite().all()))
        optimizer.eval()
        q = torch.nn.Parameter(p.detach().clone())
        continued = ScheduleFreeAdamW([q], lr=0.9)
        continued.load_state_dict(copy.deepcopy(optimizer.state_dict()))
        self.assertFalse(continued.param_groups[0]["training"])
        for solver, parameter in ((optimizer, p), (continued, q)):
            solver.train()
            parameter.grad = torch.ones_like(parameter)
            solver.step()
        self.assertTrue(torch.equal(p, q))

    def test_optimistic_uses_old_peak_two_projections_and_one_joint_ball(self) -> None:
        """Check optimistic uses old peak two projections and one joint ball."""
        p = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))
        q = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))
        optimizer = ClippedOptimistic([p, q], radius=1.0)
        previous, auxiliary, peak, energy = [0.0, 0.0], [0.0, 0.0], 0.0, 0.0

        def project(x: float) -> float:
            """Evaluate the scalar ball projection independently."""
            norm = math.sqrt(sum(a * a for a in x))
            return [a / max(1.0, norm) for a in x]

        for gradient in ([0.0, 0.0], [3.0, 4.0], [3.0, 4.0], [-2.0, 1.0], [0.0, 0.0]):
            difference = [g - old for g, old in zip(gradient, previous, strict=False)]
            norm = math.sqrt(sum(a * a for a in difference))
            fraction = min(peak, norm) / norm if norm else 0.0
            clipped = [
                old + fraction * delta for old, delta in zip(previous, difference, strict=False)
            ]
            peak = max(peak, norm)
            energy += sum((c - old) ** 2 for c, old in zip(clipped, previous, strict=False))
            eta = 1 / math.sqrt(peak * peak + energy) if peak * peak + energy else 0.0
            auxiliary = project([a - eta * c for a, c in zip(auxiliary, clipped, strict=False)])
            query = project([a - eta * g for a, g in zip(auxiliary, gradient, strict=False)])
            previous = list(gradient)
            p.grad, q.grad = p.new_tensor([gradient[0]]), q.new_tensor([gradient[1]])
            optimizer.step()
            torch.testing.assert_close(
                torch.cat([p, q]), p.new_tensor(query), atol=1e-14, rtol=1e-14
            )
            self.assertLessEqual(float(torch.cat([p, q]).detach().norm()), 1 + 1e-14)
            self.assertAlmostEqual(float(optimizer.state["optimistic"]["energy"]), energy)
        with self.assertRaises(ValueError):
            ClippedOptimistic([torch.nn.Parameter(torch.ones(1))])

    def test_optimistic_zero_prefix_frozen_lean_case_and_zero_radius(self) -> None:
        """Check optimistic zero prefix frozen lean case and zero radius."""
        for radius, final in ((1.0, -1.0), (0.0, 0.0)):
            p = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))
            optimizer = ClippedOptimistic([p], radius=radius)
            for g in (0.0, 1.0):
                p.grad = p.new_tensor([g])
                optimizer.step()
            self.assertEqual(float(p.detach()), final)


if __name__ == "__main__":
    unittest.main()
