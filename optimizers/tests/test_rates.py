"""Scalar rate overrides, live rate types and checkpoint continuation."""

from __future__ import annotations

import copy
import unittest

import torch

import lean_optim as opt


class RateTests(unittest.TestCase):
    """Keep float overrides equivalent to lr while preserving Tensor-rate kernels."""

    def test_float_rate_overrides_match_the_ordinary_learning_rate(self) -> None:
        """Exercise every direction family, including eager eigendecomposition."""
        for name in (
            "SGD",
            "Adam",
            "AdamWDirection",
            "AMSGrad",
            "AMSGradW",
            "AdamX",
            "AdaGrad",
            "AdamNC",
            "RMSProp",
            "Muon",
            "Dash",
            "AdaFisher",
        ):
            with self.subTest(rule=name):
                model = torch.nn.Linear(3, 2, bias=False, dtype=torch.float64)
                control = copy.deepcopy(model)
                options = {"solver": opt.EVD()} if name == "Dash" else {}
                actual = getattr(opt, name)(model, lr=0.5, rate=0.125, **options)
                expected = getattr(opt, name)(control, lr=0.125, **options)
                self.assertFalse(actual.capturable)
                for gradient in (0.125, -0.0625, 0.0):
                    model.weight.grad = torch.full_like(model.weight, gradient)
                    control.weight.grad = torch.full_like(control.weight, gradient)
                    actual.step()
                    expected.step()
                    torch.testing.assert_close(model.weight, control.weight, rtol=0, atol=0)
                if name == "AdaFisher":
                    actual.close()
                    expected.close()

    def test_live_rate_type_changes_and_float_checkpoints_keep_working(self) -> None:
        """Read the current group value rather than a stale construction-time flag."""
        p = torch.nn.Parameter(torch.ones(2, dtype=torch.float64))
        optimizer = opt.SGD([p], rate=0.125)
        p.grad = torch.ones_like(p)
        for rate in (torch.tensor(0.25), 0.125):
            optimizer.param_groups[0]["lr"] = rate
            optimizer.step()
        torch.testing.assert_close(p, torch.full_like(p, 0.625))
        restored = opt.SGD([p], rate=torch.tensor(0.5))
        restored.load_state_dict(copy.deepcopy(optimizer.state_dict()))
        restored.step()
        torch.testing.assert_close(p, torch.full_like(p, 0.5))


if __name__ == "__main__":
    unittest.main()
