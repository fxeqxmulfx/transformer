"""Independent arithmetic translations of the cited Lean recurrence definitions.

Expected values are calculated here from scalar equations, never by calling
the implementation's update helpers. Historical lab trajectories are tested
separately against their bundled immutable sources.
"""

from __future__ import annotations

import copy
import io
import math
import unittest

import torch
from torch import Tensor

import lean_optim as opt


def _coordinate_options(name: str) -> dict[str, object]:
    kwargs = {"lr": 0.2}
    if name in ("Adam", "AMSGrad", "AMSGradW", "AdamX"):
        kwargs.update(
            betas=(0.6, 0.75),
            eps=0.125,
            beta1_decay=opt.Inverse(),
            lr_decay=opt.InverseSqrt(),
        )
    if name in ("AdaGrad", "AdamNC"):
        kwargs.update(eps=0.125)
    if name == "AdamNC":
        kwargs["beta1"] = 0.6
    if name == "RMSProp":
        kwargs.update(beta2=0.75, eps=0.125)
    if name in ("SGD", "AMSGradW"):
        kwargs["weight_decay"] = 0.3
    return kwargs


class CoordinateTests(unittest.TestCase):
    """Numerical checks for coordinate."""

    def test_raw_moments_maxima_clocks_and_scheduled_coefficients(self) -> None:
        """Check raw moments maxima clocks and scheduled coefficients."""
        gradients = ([0.0, 2.0, -1.0], [3.0, -0.5, 0.0], [-0.25, 0.0, 4.0], [0.0, 0.0, 0.0])
        for dtype in (torch.float32, torch.float64):
            for name in (
                "SGD",
                "Adam",
                "AMSGrad",
                "AMSGradW",
                "AdamX",
                "AdaGrad",
                "AdamNC",
                "RMSProp",
            ):
                with self.subTest(dtype=dtype, rule=name):
                    p = torch.nn.Parameter(torch.tensor([1.0, -2.0, 0.5], dtype=dtype))
                    beta, beta2, rate, eps, wd = 0.6, 0.75, 0.2, 0.125, 0.3
                    kwargs = _coordinate_options(name)
                    optimizer = getattr(opt, name)([p], **kwargs)
                    x, m, v, maximum = p.tolist(), [0.0] * 3, [0.0] * 3, [0.0] * 3
                    for t, gradient in enumerate(gradients, 1):
                        for i, g in enumerate(gradient):
                            current = beta / t
                            if name != "RMSProp":
                                m[i] = current * m[i] + (1 - current) * g
                            v[i] = (
                                ((t - 1) * v[i] + g * g) / t
                                if name in ("AdaGrad", "AdamNC")
                                else beta2 * v[i] + (1 - beta2) * g * g
                            )
                            if name == "AdamX" and t > 1:
                                maximum[i] *= ((1 - current) / (1 - beta / (t - 1))) ** 2
                            maximum[i] = max(maximum[i], v[i])
                            denominator = (
                                math.sqrt(
                                    maximum[i] if name in ("AMSGrad", "AMSGradW", "AdamX") else v[i]
                                )
                                + eps
                            )
                            numerator = g if name in ("AdaGrad", "RMSProp") else m[i]
                            direction = g if name == "SGD" else numerator / denominator
                            if name in ("SGD", "AMSGradW"):
                                direction += wd * x[i]
                            if name not in ("SGD", "RMSProp"):
                                direction /= math.sqrt(t)
                            x[i] -= rate * direction
                        p.grad = torch.tensor(gradient, dtype=dtype)
                        optimizer.step()
                        torch.testing.assert_close(
                            p,
                            p.new_tensor(x),
                            rtol=2e-6 if dtype == torch.float32 else 2e-14,
                            atol=2e-6 if dtype == torch.float32 else 2e-14,
                        )
                        state = optimizer.state[p]
                        expected_buffers = {"m": m, "v": v, "maximum": maximum}
                        for key in state.keys() & expected_buffers.keys():
                            torch.testing.assert_close(
                                state[key], p.new_tensor(expected_buffers[key])
                            )

    def test_native_adamw_and_its_stage_direction_match_the_real_recurrence(self) -> None:
        """Check native adamw and its stage direction match the real recurrence."""
        p = torch.nn.Parameter(torch.tensor([2.0, -1.0], dtype=torch.float64))
        q = torch.nn.Parameter(p.detach().clone())
        native = opt.AdamW([p], lr=0.1, betas=(0.5, 0.75), eps=0.25, weight_decay=0.2)
        direction = opt.AdamWDirection([q], lr=0.1, betas=(0.5, 0.75), eps=0.25, weight_decay=0.2)
        x, m, v = [2.0, -1.0], [0.0, 0.0], [0.0, 0.0]
        for t, gradient in enumerate(([2.0, -1.0], [0.0, 3.0], [-4.0, 0.0]), 1):
            for i, g in enumerate(gradient):
                m[i] = 0.5 * m[i] + 0.5 * g
                v[i] = 0.75 * v[i] + 0.25 * g * g
                x[i] = 0.98 * x[i] - 0.1 * (m[i] / (1 - 0.5**t)) / (
                    math.sqrt(v[i] / (1 - 0.75**t)) + 0.25
                )
            p.grad = p.new_tensor(gradient)
            q.grad = q.new_tensor(gradient)
            native.step()
            direction.step()
            torch.testing.assert_close(p, p.new_tensor(x), atol=1e-14, rtol=1e-14)
            torch.testing.assert_close(q, p, atol=1e-14, rtol=1e-14)

    def test_checkpoint_restores_rates_moments_and_schedule_settings(self) -> None:
        """Check checkpoint restores rates moments and schedule settings."""
        p = torch.nn.Parameter(torch.tensor([1.0, -2.0], dtype=torch.float64))
        original = opt.AdamX(
            [p], lr=0.2, betas=(0.5, 0.75), beta1_decay=opt.Geometric(0.8), eps=0.1
        )
        p.grad = p.new_tensor([2.0, -1.0])
        original.step()
        stream = io.BytesIO()
        torch.save(original.state_dict(), stream)
        stream.seek(0)
        restored_state = torch.load(stream, weights_only=True)
        q = torch.nn.Parameter(p.detach().clone())
        continued = opt.AdamX([q], lr=0.9, betas=(0.1, 0.2), eps=0.8)
        continued.load_state_dict(restored_state)
        for gradient in ([0.0, 3.0], [-2.0, 1.0]):
            p.grad, q.grad = p.new_tensor(gradient), q.new_tensor(gradient)
            original.step()
            continued.step()
            self.assertTrue(torch.equal(p, q))
        self.assertEqual(continued.spec.beta1_decay, opt.Geometric(0.8))

    def test_restored_nonconstant_schedule_keeps_counting_from_constant_constructor(self) -> None:
        """Check restored nonconstant schedule keeps counting from constant constructor."""
        p = torch.nn.Parameter(torch.ones(1, dtype=torch.float64))
        original = opt.AMSGradW([p], lr=0.2, lr_decay=opt.InverseSqrt())
        p.grad = torch.ones_like(p)
        original.step()
        q = torch.nn.Parameter(p.detach().clone())
        continued = opt.AMSGradW([q])
        continued.load_state_dict(copy.deepcopy(original.state_dict()))
        p.grad, q.grad = torch.ones_like(p), torch.ones_like(q)
        original.step()
        continued.step()
        self.assertTrue(torch.equal(p, q))
        self.assertEqual(float(continued.state[q]["step"]), 2.0)

    def test_amsgrad_allows_the_source_beta_two_endpoint(self) -> None:
        """Check amsgrad allows the source beta two endpoint."""
        p = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))
        optimizer = opt.AMSGrad([p], lr=0.1, betas=(0.0, 1.0), eps=0.5)
        p.grad = torch.ones_like(p)
        optimizer.step()
        self.assertEqual(float(p.detach()), -0.2)

    def test_closure_scheduler_missing_gradients_and_dense_contract(self) -> None:
        """Check closure scheduler missing gradients and dense contract."""
        p = torch.nn.Parameter(torch.tensor(2.0))
        skipped = torch.nn.Parameter(torch.tensor(7.0))
        optimizer = opt.SGD([p, skipped], lr=0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

        def closure() -> Tensor:
            """Recompute the differentiable loss for the optimizer closure."""
            optimizer.zero_grad()
            loss = p.square()
            loss.backward()
            return loss

        self.assertEqual(float(optimizer.step(closure).detach()), 4.0)
        self.assertEqual(float(skipped.detach()), 7.0)
        scheduler.step()
        self.assertEqual(optimizer.param_groups[0]["lr"], 0.05)
        with self.assertRaises(ValueError):
            optimizer.add_param_group({"params": [torch.nn.Parameter(torch.ones(1))]})

    def test_clipping_precedes_the_moment_update_and_guard_uses_joint_norm(self) -> None:
        """Check clipping precedes the moment update and guard uses joint norm."""
        p = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))
        optimizer = opt.Clipped(opt.Adam([p], lr=0.2, betas=(0.0, 0.0)), norm=1.0)
        p.grad = p.new_tensor([3.0, 4.0])
        optimizer.step()
        torch.testing.assert_close(optimizer.state[p]["m"], p.new_tensor([3.0, 4.0]) / (5 + 1e-6))
        for sigma, expected in ((0.5, True), (1.0, False)):
            q = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
            guarded = opt.Guarded(opt.SGD([q], lr=0.1, weight_decay=0.2), sigma)
            q.grad = q.new_tensor([-1.0, -2.0])
            guarded.step()
            self.assertEqual(guarded.report()["guard"]["accepted"], int(expected))
            torch.testing.assert_close(q, q.new_tensor([1.08, 2.16] if expected else [1.1, 2.2]))


if __name__ == "__main__":
    unittest.main()
