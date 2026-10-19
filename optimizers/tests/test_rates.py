"""Scalar rate overrides, live rate types and checkpoint continuation."""

from __future__ import annotations

import copy
import math
import unittest
from itertools import product

import torch

import lean_optim as opt

_DIRECTION_RULES = (
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
)
_TENSOR_RATE_RULES = (*_DIRECTION_RULES, "AMSGradMD", "AMSGradMDW", "AdamMDW")


class RateTests(unittest.TestCase):
    """Keep float overrides equivalent to lr while preserving Tensor-rate kernels."""

    def test_float_rate_overrides_match_the_ordinary_learning_rate(self) -> None:
        """Exercise every direction family, including eager eigendecomposition."""
        for name in _DIRECTION_RULES:
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

    def test_invalid_float_overrides_fail_before_mutation_or_hook_registration(self) -> None:
        """Reject nonfinite and negative overrides across every public rate constructor."""
        names = (*_TENSOR_RATE_RULES, "ScheduleFreeAMSGradMDW", "ScheduleFreeAdamMDW")
        for name, rate in product(names, (-0.125, math.nan, math.inf, -math.inf)):
            with self.subTest(rule=name, rate=rate):
                model = torch.nn.Linear(3, 2, bias=False, dtype=torch.float64)
                before = model.weight.detach().clone()
                with self.assertRaisesRegex(ValueError, "rate"):
                    getattr(opt, name)(model, lr=0.125, rate=rate)
                torch.testing.assert_close(model.weight, before, rtol=0, atol=0)
                self.assertFalse(model._forward_hooks)  # noqa: SLF001 - Check failed constructor cleanup.

    def test_invalid_tensor_overrides_fail_before_state_or_hooks_are_created(self) -> None:
        """Reject invalid values, non-scalar storage and complex rates without copying them."""
        rates = (
            *(torch.tensor(value) for value in (-0.125, math.nan, math.inf, -math.inf)),
            torch.empty(0),
            torch.tensor([0.125, 0.25]),
            torch.tensor(0.125 + 0j),
        )
        for name, rate in product(_TENSOR_RATE_RULES, rates):
            with self.subTest(rule=name, rate=rate):
                model = torch.nn.Linear(3, 2, bias=False, dtype=torch.float64)
                before = model.weight.detach().clone()
                with self.assertRaisesRegex(ValueError, "rate"):
                    getattr(opt, name)(model, lr=0.125, rate=rate)
                torch.testing.assert_close(model.weight, before, rtol=0, atol=0)
                self.assertFalse(model._forward_hooks)  # noqa: SLF001 - Check failed constructor cleanup.

    def test_valid_zero_and_positive_tensor_rates_keep_caller_storage(self) -> None:
        """Keep live scalar tensors usable by all direction and ordinary MD constructors."""
        for name, value in product(_TENSOR_RATE_RULES, (0.0, 0.125)):
            with self.subTest(rule=name, rate=value):
                model = torch.nn.Linear(3, 2, bias=False, dtype=torch.float64)
                control = copy.deepcopy(model)
                rate = torch.tensor(value, dtype=torch.float64)
                actual = getattr(opt, name)(model, lr=0.5, rate=rate)
                expected = getattr(opt, name)(control, lr=value)
                model.weight.grad = torch.full_like(model.weight, 0.125)
                control.weight.grad = torch.full_like(control.weight, 0.125)
                actual.step()
                expected.step()
                self.assertIs(actual.param_groups[0]["lr"], rate)
                self.assertEqual(float(rate), value)
                torch.testing.assert_close(model.weight, control.weight, rtol=1e-13, atol=1e-13)
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

    def test_tensor_rate_shapes_update_matrices_and_scalars_and_survive_loading(self) -> None:
        """Keep every one-element shape equivalent to a live scalar, including guarded MD."""
        names = (*_TENSOR_RATE_RULES, "AMSGradMD-guard")
        for name, shape in product(names, ((), (1,), (1, 1), (1, 1, 1))):
            with self.subTest(rule=name, shape=shape):
                model = torch.nn.Linear(3, 2, bias=False, dtype=torch.float64)
                model.register_parameter("scale", torch.nn.Parameter(torch.tensor(0.5).double()))
                control = copy.deepcopy(model)
                rate = torch.full(shape, 0.125, dtype=torch.float64)
                scalar = torch.tensor(0.125, dtype=torch.float64)
                cls = getattr(opt, name.removesuffix("-guard"))
                options = {"guard_sigma": 0.5} if name.endswith("-guard") else {}
                actual, expected = (
                    cls(model, rate=rate, **options),
                    cls(control, rate=scalar, **options),
                )
                for value, gradient in ((0.0, 0.125), (0.125, -0.0625), (0.25, 0.0)):
                    rate.fill_(value)
                    scalar.fill_(value)
                    for parameter in (*model.parameters(), *control.parameters()):
                        parameter.grad = torch.full_like(parameter, gradient)
                    actual.step()
                    expected.step()
                    for parameter, reference in zip(
                        model.parameters(), control.parameters(), strict=True
                    ):
                        torch.testing.assert_close(parameter, reference, rtol=0, atol=0)
                    saved = copy.deepcopy(actual.state_dict())
                    actual.load_state_dict(saved)
                    expected.load_state_dict(copy.deepcopy(saved))
                    self.assertIs(actual.param_groups[0]["lr"], rate)
                    self.assertIs(expected.param_groups[0]["lr"], scalar)
                    self.assertEqual(rate.shape, shape)
                if name == "AdaFisher":
                    actual.close()
                    expected.close()


if __name__ == "__main__":
    unittest.main()
