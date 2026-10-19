"""Independent-rate references and ordinary PyTorch scheduler compatibility."""

from __future__ import annotations

import io
import unittest

import torch
from _warmup_cases import factories, gradients, parameters

import lean_optim as opt


def _assert_state(test: unittest.TestCase, actual: object, expected: object) -> None:
    if isinstance(actual, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(actual, dict):
        test.assertEqual(actual.keys(), expected.keys())
        for key, value in actual.items():
            _assert_state(test, value, expected[key])
    else:
        test.assertEqual(actual, expected)


class WarmupTests(unittest.TestCase):
    """Check every MD rate, exact zero freezes, scheduler clocks and checkpoints."""

    def test_torch_scheduler_scales_all_md_rates_against_independent_callbacks(self) -> None:
        """Match manually supplied per-step rates, including an explicit direction rate."""
        factors = (0.0, 0.0, 0.25, 1.0, 0.5, 0.0, 0.75, 1.0)
        for name, factory in factories().items():
            with self.subTest(rule=name):
                source, control = parameters(), parameters()
                options = {"lr": 0.125, "gain_rate": 0.03125, "auxiliary_rate": 0.015625}
                if name == "SFMD":
                    options["warmup"] = 3
                scheduled, manual = factory(source, **options), factory(control, **options)
                peaks = {
                    key: manual.param_groups[0][key]
                    for key in ("lr", "direction_rate", "gain_rate", "auxiliary_rate")
                    if key in manual.param_groups[0]
                }
                scheduler = torch.optim.lr_scheduler.LambdaLR(
                    scheduled, lambda index: factors[min(index, len(factors) - 1)]
                )
                for index, factor in enumerate(factors):
                    for key, peak in peaks.items():
                        manual.param_groups[0][key] = None if peak is None else peak * factor
                    before = [p.detach().clone() for p in source]
                    for optimizer in (scheduled, manual):
                        gradients(optimizer, (-1) ** index / 8)
                        optimizer.step()
                    for p, q, old in zip(source, control, before, strict=True):
                        torch.testing.assert_close(p, q, rtol=0, atol=0)
                        _assert_state(self, scheduled.state[p], manual.state[q])
                        if factor == 0:
                            self.assertTrue(torch.equal(p, old))
                    scheduler.step()

    def test_linear_warmup_supports_the_package_rules_and_native_parameter_groups(self) -> None:
        """Exercise a zero start and exact linear rates across the exported LR rules."""
        names = (
            "SGD",
            "Adam",
            "AMSGrad",
            "AMSGradW",
            "AdamX",
            "AdaGrad",
            "AdamNC",
            "RMSProp",
            "AdamW",
            "AdamWDirection",
            "Muon",
            "Dash",
            "AMSGradMD",
            "ScheduleFreeSGD",
            "ScheduleFreeAdamW",
            "ScheduleFreeAMSGradW",
            "ScheduleFreeAMSGradMD",
        )
        for name in names:
            with self.subTest(rule=name):
                values = parameters()
                optimizer = getattr(opt, name)(values, lr=0.125)
                self._linear_case(optimizer)
        model = torch.nn.Linear(3, 2, bias=False, dtype=torch.float64)
        optimizer = opt.AdaFisher(model, lr=0.125)
        self._linear_case(optimizer)
        optimizer.close()
        for wrapper in (opt.Guarded, opt.Clipped, lambda base: opt.Magma(base, updates=10)):
            self._linear_case(wrapper(opt.Adam(parameters(), lr=0.125)))
        values = parameters()
        native = opt.standard.AdamW(
            [{"params": [values[0]], "lr": 0.125}, {"params": [values[1]], "lr": 0.0625}]
        )
        self._linear_case(native)

    def _linear_case(self, optimizer: torch.optim.Optimizer) -> None:
        scheduler = opt.LinearWarmup(optimizer, 4)
        peaks = [group["initial_lr"] for group in optimizer.param_groups]
        for factor in (0.0, 0.25, 0.5, 0.75, 1.0, 1.0):
            self.assertEqual(scheduler.get_last_lr(), [peak * factor for peak in peaks])
            before = [
                p.detach().clone() for group in optimizer.param_groups for p in group["params"]
            ]
            for group in optimizer.param_groups:
                for p in group["params"]:
                    p.grad = torch.full_like(p, 0.125)
            optimizer.step()
            after = [p for group in optimizer.param_groups for p in group["params"]]
            if factor == 0:
                self.assertTrue(
                    all(torch.equal(p, old) for p, old in zip(after, before, strict=True))
                )
            self.assertTrue(all(bool(p.isfinite().all()) for p in after))
            scheduler.step()

    def test_checkpoint_resumes_the_schedule_and_component_rates(self) -> None:
        """Round-trip plain scheduling state and optimizer state through torch.save."""
        for name, factory in factories().items():
            with self.subTest(rule=name):
                original = factory(parameters(), lr=0.125)
                schedule = opt.LearningRateSchedule(
                    original, warmup=4, total_steps=12, decay="cosine", final_factor=0.125
                )
                for _ in range(6):
                    gradients(original, 0.125)
                    original.step()
                    schedule.step()
                stream = io.BytesIO()
                torch.save((original.state_dict(), schedule.state_dict()), stream)
                stream.seek(0)
                optimizer_state, scheduler_state = torch.load(stream, weights_only=True)
                restored = factory(parameters(), lr=0.5)
                continued = opt.LearningRateSchedule(restored, warmup=17)
                with torch.no_grad():
                    for p, q in zip(
                        original.param_groups[0]["params"],
                        restored.param_groups[0]["params"],
                        strict=True,
                    ):
                        q.copy_(p)
                restored.load_state_dict(optimizer_state)
                continued.load_state_dict(scheduler_state)
                for _ in range(5):
                    for optimizer, scheduler in ((original, schedule), (restored, continued)):
                        gradients(optimizer, -0.25)
                        optimizer.step()
                        scheduler.step()
                    self.assertEqual(schedule.get_last_lr(), continued.get_last_lr())
                    for p, q in zip(
                        original.param_groups[0]["params"],
                        restored.param_groups[0]["params"],
                        strict=True,
                    ):
                        self.assertTrue(torch.equal(p, q))
                        _assert_state(self, original.state[p], restored.state[q])

    def test_independent_rates_remain_available_without_a_scheduler(self) -> None:
        """A deliberately zero direction rate can still leave configured gains active."""
        for factory in (opt.AMSGradMD, opt.ScheduleFreeAMSGradMD):
            values = parameters()
            optimizer = factory(values, lr=0.0)
            before = [p.detach().clone() for p in values]
            gradients(optimizer, 0.125)
            optimizer.step()
            self.assertTrue(
                all(not torch.equal(p, old) for p, old in zip(values, before, strict=True))
            )
            frozen = factory(parameters(), lr=0.0)
            opt.LinearWarmup(frozen, 4)
            old = [p.detach().clone() for p in frozen.param_groups[0]["params"]]
            gradients(frozen, 0.125)
            frozen.step()
            self.assertTrue(
                all(
                    torch.equal(p, q)
                    for p, q in zip(frozen.param_groups[0]["params"], old, strict=True)
                )
            )

    def test_validation_disabled_warmup_and_explicit_completed_counts(self) -> None:
        """Reject unsupported schedules before changing optimizer hyperparameters."""
        for value in (-1, True, 1.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                opt.LinearWarmup(opt.SGD(parameters()), value)
        for optimizer in (
            object(),
            opt.ClippedOptimistic([torch.nn.Parameter(torch.zeros(1))]),
            opt.standard.Rprop(parameters()),
        ):
            with self.assertRaises(TypeError):
                opt.LinearWarmup(optimizer, 4)
        with self.assertRaises(ValueError):
            opt.LinearWarmup(opt.ScheduleFreeAMSGradMD(parameters(), warmup=3), 4)
        optimizer = opt.SGD(parameters(), lr=0.125)
        scheduler = opt.LinearWarmup(optimizer, 0)
        self.assertEqual(scheduler.get_last_lr(), [0.125])
        scheduler = opt.LinearWarmup(optimizer, 4)
        scheduler.step(2)
        self.assertEqual(scheduler.get_last_lr(), [0.0625])
        rates = scheduler.get_last_lr()
        rates[0] = 9.0
        self.assertEqual(optimizer.param_groups[0]["lr"], 0.0625)
        with self.assertRaises(ValueError):
            scheduler.step(-1)

    def test_guard_can_fall_back_when_proposal_component_rates_are_zero(self) -> None:
        """Keep the checked method's true-gradient fallback at its positive outer rate."""
        values = parameters()
        optimizer = opt.AMSGradMD(
            values,
            lr=0.125,
            direction_rate=0.0625,
            gain_rate=0.03125,
            auxiliary_rate=0.015625,
            guard_sigma=0.5,
        )
        optimizer.param_groups[0].update(direction_rate=0.0, gain_rate=0.0, auxiliary_rate=0.0)
        scheduler = opt.LearningRateSchedule(optimizer)
        before = [p.detach().clone() for p in values]
        gradients(optimizer, 0.125)
        optimizer.step()
        scheduler.step()
        for p, old in zip(values, before, strict=True):
            torch.testing.assert_close(p, old - 0.015625, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
