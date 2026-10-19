"""Physical MDW recurrences, native auxiliary updates and singular storage cases."""

from __future__ import annotations

import unittest
from itertools import product

import torch
from _mdw_reference import Configuration, History, Reference, fuse

import lean_optim as opt

_FACTORS = (0.0, 0.25, 1.0, 0.5, 0.0, 0.75, 1.0)


def _values(dtype: torch.dtype = torch.float64) -> list[torch.Tensor]:
    return [
        torch.nn.Parameter(torch.tensor([[0.5, -0.25, 0.125], [0.25, 0.75, -0.5]], dtype=dtype)),
        torch.nn.Parameter(torch.tensor([0.25, -0.125], dtype=dtype)),
    ]


class MDWTests(unittest.TestCase):
    """Compare every history and physical iterate to independently differentiated factors."""

    def _history(
        self,
        actual: dict[str, object],
        expected: History,
        configuration: Configuration,
        prefix: str = "",
    ) -> None:
        tolerance = 3e-5 if expected.base.dtype == torch.float32 else 2e-12
        key = "second" if configuration.free else "v"
        torch.testing.assert_close(
            actual[prefix + key], expected.second, rtol=tolerance, atol=tolerance
        )
        if not configuration.free:
            torch.testing.assert_close(
                actual[prefix + "m"], expected.moment, rtol=tolerance, atol=tolerance
            )
        self.assertEqual(prefix + "maximum" in actual, not configuration.adam)
        if not configuration.adam:
            torch.testing.assert_close(
                actual[prefix + "maximum"], expected.maximum, rtol=tolerance, atol=tolerance
            )
        if configuration.free:
            for name in ("base", "evaluation"):
                torch.testing.assert_close(
                    actual[name], getattr(expected, name), rtol=tolerance, atol=tolerance
                )
            self.assertAlmostEqual(actual["total"], expected.total)
            self.assertEqual(actual["step"], expected.step)

    def test_four_variants_match_autograd_physical_decay_and_all_histories(self) -> None:
        """Check warmup, decay, zero rates, moments and evaluation at each update."""
        names = ("AMSGradMDW", "AdamMDW", "ScheduleFreeAMSGradMDW", "ScheduleFreeAdamMDW")
        for name, dtype in product(names, (torch.float32, torch.float64)):
            free, adam = name.startswith("ScheduleFree"), "AMSGrad" not in name
            configuration = Configuration(adam, free, warmup=3 if free else 0)
            with self.subTest(rule=name, dtype=dtype):
                values = _values(dtype)
                options = {"warmup": configuration.warmup} if free else {}
                optimizer = getattr(opt, name)(
                    values,
                    lr=0.0625,
                    gain_rate=0.03125,
                    auxiliary_rate=0.015625,
                    betas=(0.5, 0.75),
                    eps=0.125,
                    weight_decay=0.25,
                    **options,
                )
                reference = Reference(values, configuration)
                scheduler = torch.optim.lr_scheduler.LambdaLR(
                    optimizer, lambda t: _FACTORS[min(t, len(_FACTORS) - 1)]
                )
                for index, factor in enumerate(_FACTORS):
                    gradients = [
                        torch.linspace(-0.25, 0.5, p.numel(), dtype=dtype).reshape_as(p)
                        * (1.0, -0.5, 0.0)[index % 3]
                        for p in values
                    ]
                    old = [p.detach().clone() for p in values]
                    for parameter, gradient in zip(values, gradients, strict=True):
                        parameter.grad = gradient.clone()
                    reference.step(
                        gradients, tuple(rate * factor for rate in (0.0625, 0.03125, 0.015625))
                    )
                    optimizer.step()
                    expected_auxiliary = (
                        reference.auxiliary.query(configuration)
                        if free
                        else reference.auxiliary.base
                    )
                    tolerance = 3e-5 if dtype == torch.float32 else 2e-12
                    for actual, expected, before, gradient in zip(
                        values, (reference.weight, expected_auxiliary), old, gradients, strict=True
                    ):
                        torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
                        torch.testing.assert_close(actual.grad, gradient, rtol=0, atol=0)
                        if factor == 0:
                            self.assertTrue(torch.equal(actual, before))
                    state = optimizer.state[values[0]]
                    if free:
                        for key, history in zip(
                            ("direction", "row", "column"), reference.factors, strict=True
                        ):
                            self._history(state[key], history, configuration)
                        optimizer.eval()
                        torch.testing.assert_close(
                            values[0],
                            fuse(reference.points(evaluation=True), smooth=True),
                            rtol=tolerance,
                            atol=tolerance,
                        )
                        optimizer.train()
                    else:
                        for prefix, history in zip(
                            ("", "row.", "column."), reference.factors, strict=True
                        ):
                            self._history(state, history, configuration, prefix)
                        if adam:
                            self.assertEqual(float(state["step"]), index + 1)
                    self._history(optimizer.state[values[1]], reference.auxiliary, configuration)
                    scheduler.step()

    def test_auxiliary_adammdw_matches_native_adamw(self) -> None:
        """Excluded matrices and vectors retain native bias corrections and old-weight decay."""
        actual = torch.nn.Parameter(torch.tensor([0.5, -0.25], dtype=torch.float64))
        expected = torch.nn.Parameter(actual.detach().clone())
        optimizer = opt.AdamMDW(
            [actual],
            lr=0.125,
            auxiliary_rate=0.03125,
            matrix_params=[],
            betas=(0.5, 0.75),
            eps=0.125,
            weight_decay=0.25,
        )
        native = opt.standard.AdamW(
            [expected], lr=0.03125, betas=(0.5, 0.75), eps=0.125, weight_decay=0.25
        )
        for gradient in ([0.25, -0.5], [0.0, 0.125], [-0.25, 0.0]):
            actual.grad, expected.grad = actual.new_tensor(gradient), expected.new_tensor(gradient)
            optimizer.step()
            native.step()
            torch.testing.assert_close(actual, expected, rtol=1e-14, atol=1e-14)

    def test_zero_decay_retains_the_existing_amsgrad_md_recurrence(self) -> None:
        """Keep both old AMSGrad variants bit-identical when decay is disabled."""
        for base, extended in (
            (opt.AMSGradMD, opt.AMSGradMDW),
            (opt.ScheduleFreeAMSGradMD, opt.ScheduleFreeAMSGradMDW),
        ):
            values, expected = _values(), _values()
            optimizer, control = extended(values, weight_decay=0.0), base(expected)
            for gradient in (0.125, -0.0625, 0.0):
                for parameter in (*values, *expected):
                    parameter.grad = torch.full_like(parameter, gradient)
                optimizer.step()
                control.step()
                for actual, reference in zip(values, expected, strict=True):
                    torch.testing.assert_close(actual, reference, rtol=0, atol=0)

    def test_zero_physical_result_keeps_finite_gains_and_can_continue(self) -> None:
        """The zero-result case is explicit in Lean and does not store log zero."""
        for name in ("AMSGradMDW", "AdamMDW", "ScheduleFreeAMSGradMDW", "ScheduleFreeAdamMDW"):
            with self.subTest(rule=name):
                p = torch.nn.Parameter(torch.tensor([[2.0]], dtype=torch.float64))
                optimizer = getattr(opt, name)(
                    [p], lr=0.25, gain_rate=0.125, betas=(0.5, 0.0), eps=1.0, weight_decay=4.0
                )
                p.grad = torch.zeros_like(p)
                optimizer.step()
                torch.testing.assert_close(p, torch.zeros_like(p), rtol=0, atol=1e-14)
                p.grad.fill_(0.125)
                optimizer.step()
                self.assertTrue(bool(p.isfinite().all()))
                state = optimizer.state[p]
                rows = state["row"]
                self.assertTrue(
                    bool((rows["base"] if isinstance(rows, dict) else rows).isfinite().all())
                )


if __name__ == "__main__":
    unittest.main()
