"""Independent recurrences and latent-query autograd checks for the new variants.

Lean definitions: Transformer.ScheduleFree.AMSGrad.step, mdStep and mdRun.
The MD reference differentiates the fused query with PyTorch autograd; it
does not reuse the optimizer's manually derived factor gradients.
"""

from __future__ import annotations

import copy
import math
import unittest
from dataclasses import dataclass

import torch

from lean_optim import ScheduleFreeAMSGradMD, ScheduleFreeAMSGradW


@dataclass
class _Reference:
    """Allocating reference iterates and raw second-moment histories."""

    base: torch.Tensor
    evaluation: torch.Tensor
    second: torch.Tensor
    maximum: torch.Tensor
    total: float = 0.0
    step: int = 0

    @classmethod
    def initial(cls, point: torch.Tensor) -> _Reference:
        """Initialize from independently specified optimization coordinates."""
        return cls(point.clone(), point.clone(), torch.zeros_like(point), torch.zeros_like(point))

    def query(self, beta: float) -> torch.Tensor:
        """Build the oracle point with ordinary allocating arithmetic."""
        return (1 - beta) * self.base + beta * self.evaluation

    def advance(self, gradient: torch.Tensor, rate: float, *, radius: float | None = None) -> None:
        """Apply the documented reference with beta2=3/4, epsilon=.05 and warmup=3."""
        self.step += 1
        eta = rate * math.sqrt(1 - 0.75**self.step) * min(1, self.step / 3)
        self.second = 0.75 * self.second + 0.25 * gradient.square()
        self.maximum = torch.maximum(self.maximum, self.second)
        self.base = self.base - eta * gradient / (self.maximum.sqrt() + 0.05)
        if radius is not None and eta:
            norm = float(self.base.norm())
            self.base = self.base * (radius / norm) if norm else torch.zeros_like(self.base)
        self.total += eta**2
        coefficient = eta**2 / self.total if self.total else 0.0
        self.evaluation = (1 - coefficient) * self.evaluation + coefficient * self.base


def _fuse(factors: list[torch.Tensor]) -> torch.Tensor:
    direction, row, column = factors
    return (
        direction
        * torch.logaddexp(row, torch.zeros_like(row))[:, None]
        * torch.logaddexp(column, torch.zeros_like(column))[None, :]
    )


class ScheduleFreeAMSGradTests(unittest.TestCase):
    """Check raw maxima, genuine query gradients and the complete optimizer lifecycle."""

    def test_w_maximum_query_decay_warmup_and_changing_rates(self) -> None:
        """Compare all buffers with scalar arithmetic through zero-rate prefixes."""
        for beta in (0.0, 0.6, 0.95):
            for warmup in (0, 3):
                with self.subTest(beta=beta, warmup=warmup):
                    p = torch.nn.Parameter(torch.tensor([2.0, -1.0], dtype=torch.float64))
                    optimizer = ScheduleFreeAMSGradW(
                        [p],
                        lr=0.2,
                        betas=(beta, 0.75),
                        eps=0.125,
                        weight_decay=0.3,
                        warmup=warmup,
                    )
                    base, evaluation, variance, maximum = (
                        [2.0, -1.0],
                        [2.0, -1.0],
                        [0.0] * 2,
                        [0.0] * 2,
                    )
                    total = 0.0
                    gradients = ([2.0, -3.0], [0.1, 0.0], [0.0, 1.0], [-1.0, 0.2], [3.0, 0.0])
                    for t, (rate, gradient) in enumerate(
                        zip((0.0, 0.2, 0.1, 0.0, 0.3), gradients, strict=True), 1
                    ):
                        query = [
                            (1 - beta) * z + beta * x for z, x in zip(base, evaluation, strict=True)
                        ]
                        eta = rate * math.sqrt(1 - 0.75**t) * (min(1, t / warmup) if warmup else 1)
                        total += eta**2
                        coefficient = eta**2 / total if total else 0.0
                        for i, g in enumerate(gradient):
                            variance[i] = 0.75 * variance[i] + 0.25 * g * g
                            maximum[i] = max(maximum[i], variance[i])
                            base[i] -= eta * (g / (math.sqrt(maximum[i]) + 0.125) + 0.3 * query[i])
                            evaluation[i] = (1 - coefficient) * evaluation[i] + coefficient * base[
                                i
                            ]
                        optimizer.param_groups[0]["lr"] = rate
                        p.grad = p.new_tensor(gradient)
                        optimizer.step()
                        for key, expected in (
                            ("base", base),
                            ("evaluation", evaluation),
                            ("second", variance),
                            ("maximum", maximum),
                        ):
                            torch.testing.assert_close(
                                optimizer.state[p][key],
                                p.new_tensor(expected),
                                rtol=1e-13,
                                atol=1e-13,
                            )
                        self.assertAlmostEqual(optimizer.state[p]["total"], total)
                        torch.testing.assert_close(
                            p,
                            p.new_tensor(
                                [
                                    (1 - beta) * z + beta * x
                                    for z, x in zip(base, evaluation, strict=True)
                                ]
                            ),
                        )
                        optimizer.eval()
                        torch.testing.assert_close(p, p.new_tensor(evaluation))
                        optimizer.train()

    def test_w_frozen_lean_maximum_and_weighted_average_case(self) -> None:
        """Two reachable steps distinguish the raw maximum from current variance."""
        p = torch.nn.Parameter(torch.tensor([2.0], dtype=torch.float64))
        optimizer = ScheduleFreeAMSGradW([p], lr=1.0, betas=(0.5, 0.0), eps=1.0)
        for gradient in (2.0, 1.0):
            p.grad = p.new_tensor([gradient])
            optimizer.step()
        self.assertEqual(float(optimizer.state[p]["second"]), 1.0)
        self.assertEqual(float(optimizer.state[p]["maximum"]), 4.0)
        torch.testing.assert_close(optimizer.state[p]["base"], p.new_tensor([1.0]))
        torch.testing.assert_close(p, p.new_tensor([13 / 12]))
        optimizer.eval()
        torch.testing.assert_close(p, p.new_tensor([7 / 6]))

    def test_md_factor_gradients_match_autograd_at_the_actual_query(self) -> None:
        """Compare every latent buffer through distinct changing component rates."""
        for beta in (0.0, 0.6, 0.95):
            with self.subTest(beta=beta):
                p = torch.nn.Parameter(
                    torch.tensor([[0.5, -0.2, 0.3], [0.1, 0.7, -0.4]], dtype=torch.float64)
                )
                auxiliary = torch.nn.Parameter(torch.tensor([0.2, -0.1], dtype=torch.float64))
                radius = float(p.detach().norm())
                gain = math.log(math.expm1(1.0))
                references = [
                    _Reference.initial(p.detach()),
                    _Reference.initial(p.new_full((2,), gain)),
                    _Reference.initial(p.new_full((3,), gain)),
                ]
                extra = _Reference.initial(auxiliary.detach())
                optimizer = ScheduleFreeAMSGradMD(
                    [p, auxiliary],
                    lr=0.2,
                    gain_rate=0.1,
                    auxiliary_rate=0.2,
                    betas=(beta, 0.75),
                    eps=0.05,
                    warmup=3,
                )
                rates = ((0.2, 0.1, 0.2), (0.05, 0.2, 0.0), (0.0, 0.0, 0.05), (0.4, 0.1, 0.1))
                for t, (direction_rate, gain_rate, auxiliary_rate) in enumerate(rates):
                    query = [state.query(beta).detach().requires_grad_() for state in references]
                    fused = _fuse(query)
                    torch.testing.assert_close(p, fused, rtol=1e-13, atol=1e-13)
                    gradient = (
                        (torch.arange(6, dtype=p.dtype).reshape_as(p) - 2) * (-1) ** t / (t + 1)
                    )
                    gradients = torch.autograd.grad((fused * gradient).sum(), query)
                    for index, (reference, g) in enumerate(zip(references, gradients, strict=True)):
                        reference.advance(
                            g,
                            direction_rate if index == 0 else gain_rate,
                            radius=radius if index == 0 else None,
                        )
                    extra_gradient = auxiliary.new_tensor([0.1 * (t + 1), -0.3])
                    extra.advance(extra_gradient, auxiliary_rate)
                    group = optimizer.param_groups[0]
                    group.update(
                        lr=direction_rate, gain_rate=gain_rate, auxiliary_rate=auxiliary_rate
                    )
                    p.grad, auxiliary.grad = gradient, extra_gradient
                    optimizer.step()
                    torch.testing.assert_close(p.grad, gradient, rtol=0, atol=0)
                    for key, reference in zip(
                        ("direction", "row", "column"), references, strict=True
                    ):
                        for buffer in ("base", "evaluation", "second", "maximum"):
                            torch.testing.assert_close(
                                optimizer.state[p][key][buffer],
                                getattr(reference, buffer),
                                rtol=1e-12,
                                atol=1e-13,
                            )
                        self.assertAlmostEqual(optimizer.state[p][key]["total"], reference.total)
                    torch.testing.assert_close(
                        p,
                        _fuse([state.query(beta) for state in references]),
                        rtol=1e-12,
                        atol=1e-13,
                    )
                    torch.testing.assert_close(auxiliary, extra.query(beta), rtol=1e-13, atol=1e-13)
                    self.assertAlmostEqual(
                        float(optimizer.state[p]["direction"]["base"].norm()), radius
                    )
                    optimizer.eval()
                    torch.testing.assert_close(
                        p, _fuse([state.evaluation for state in references]), rtol=1e-12, atol=1e-13
                    )
                    torch.testing.assert_close(auxiliary, extra.evaluation)
                    evaluation = p.detach().clone()
                    optimizer.eval()
                    self.assertTrue(torch.equal(p, evaluation))
                    optimizer.train()

    def test_md_frozen_lean_sphere_and_factor_average_case(self) -> None:
        """A sign reversal puts the base on its sphere and its evaluation at zero."""
        p = torch.nn.Parameter(torch.tensor([[2.0]], dtype=torch.float64))
        optimizer = ScheduleFreeAMSGradMD(
            [p], lr=4.0, gain_rate=0.0, auxiliary_rate=0.0, betas=(0.5, 0.0), eps=1.0
        )
        for gradient, expected in ((2.0, -2.0), (-2.0, 1.0)):
            p.grad = p.new_tensor([[gradient]])
            optimizer.step()
            torch.testing.assert_close(p, p.new_tensor([[expected]]))
        self.assertAlmostEqual(float(optimizer.state[p]["direction"]["base"]), 2.0)
        optimizer.eval()
        torch.testing.assert_close(p, torch.zeros_like(p), atol=1e-14, rtol=0)

    def test_md_zero_direction_rate_lean_gain_case(self) -> None:
        """The raw-zero-gain checkpoint has exact chain-rule gradients of one."""
        p = torch.nn.Parameter(torch.tensor([[2.0]], dtype=torch.float64))
        optimizer = ScheduleFreeAMSGradMD(
            [p], lr=0.0, gain_rate=1.0, auxiliary_rate=0.0, betas=(0.6, 0.0), eps=1.0
        )
        for key in ("row", "column"):
            optimizer.state[p][key]["base"].zero_()
            optimizer.state[p][key]["evaluation"].zero_()
        optimizer.train()
        p.grad = p.new_tensor([[1 / math.log(2)]])
        optimizer.step()
        for key in ("row", "column"):
            torch.testing.assert_close(optimizer.state[p][key]["base"], p.new_tensor([-0.5]))
            torch.testing.assert_close(optimizer.state[p][key]["second"], p.new_tensor([1.0]))
        expected = 2 * math.log1p(math.exp(-0.5)) ** 2
        torch.testing.assert_close(p, p.new_tensor([[expected]]), rtol=1e-13, atol=1e-13)

    def test_md_smooth_gain_derivatives_match_autograd_beyond_softplus_thresholds(self) -> None:
        """Check sigmoid against the smooth gain map for large signed raw gains."""
        p = torch.nn.Parameter(torch.tensor([[0.5, -0.2], [0.1, 0.7]], dtype=torch.float64))
        optimizer = ScheduleFreeAMSGradMD([p], betas=(0.6, 0.75))
        for key, values in (("row", [25.0, -30.0]), ("column", [-25.0, 30.0])):
            for buffer in ("base", "evaluation"):
                optimizer.state[p][key][buffer].copy_(p.new_tensor(values))
        optimizer.train()
        points = [
            p.detach().new_tensor([[0.5, -0.2], [0.1, 0.7]]).requires_grad_(),
            p.new_tensor([25.0, -30.0]).requires_grad_(),
            p.new_tensor([-25.0, 30.0]).requires_grad_(),
        ]
        gradient = p.new_tensor([[0.2, -0.3], [-0.1, 0.4]])
        expected = torch.autograd.grad((_fuse(points) * gradient).sum(), points)
        p.grad = gradient
        optimizer.step()
        for key, g in zip(("direction", "row", "column"), expected, strict=True):
            torch.testing.assert_close(
                optimizer.state[p][key]["second"], 0.25 * g.square(), rtol=5e-13, atol=1e-30
            )

    def test_checkpoints_resume_in_training_and_evaluation_modes(self) -> None:
        """Restore nested factors, hyperparameters and modes without scratch state."""
        for factory in (ScheduleFreeAMSGradW, ScheduleFreeAMSGradMD):
            for evaluation in (False, True):
                with self.subTest(rule=factory.__name__, evaluation=evaluation):
                    model = torch.nn.Linear(3, 2, dtype=torch.float64)
                    optimizer = factory(model, lr=0.2, betas=(0.6, 0.75), warmup=3)
                    for gradient in (0.1, -0.2, 0.05):
                        for p in model.parameters():
                            p.grad = torch.full_like(p, gradient)
                        optimizer.step()
                    if evaluation:
                        optimizer.eval()
                    checkpoint = copy.deepcopy(optimizer.state_dict())
                    replacement = torch.nn.Linear(3, 2, dtype=torch.float64)
                    continued = factory(replacement, lr=0.9)
                    replacement.load_state_dict(copy.deepcopy(model.state_dict()))
                    continued.load_state_dict(checkpoint)
                    self.assertFalse(continued.workspace.buffers)
                    self.assertEqual(continued.param_groups[0]["training"], not evaluation)
                    if evaluation:
                        with self.assertRaises(RuntimeError):
                            optimizer.step()
                        with self.assertRaises(RuntimeError):
                            continued.step()
                    optimizer.train()
                    continued.train()
                    for _ in range(3):
                        for network, solver in ((model, optimizer), (replacement, continued)):
                            solver.zero_grad(set_to_none=True)
                            loss = sum(p.square().sum() for p in network.parameters())
                            loss.backward()
                            solver.step()
                    for p, q in zip(model.parameters(), replacement.parameters(), strict=True):
                        self.assertTrue(torch.equal(p, q))

    def test_zero_rates_freeze_factors_and_auxiliary_values(self) -> None:
        """Zero component rates preserve installed weights while maxima advance."""
        model = torch.nn.Linear(3, 2, dtype=torch.float64)
        optimizer = ScheduleFreeAMSGradMD(model, lr=0.0, gain_rate=0.0, auxiliary_rate=0.0)
        initial = [p.detach().clone() for p in model.parameters()]
        for _ in range(3):
            for p in model.parameters():
                p.grad = torch.ones_like(p)
            optimizer.step()
            for p, value in zip(model.parameters(), initial, strict=True):
                self.assertTrue(torch.equal(p, value))
        self.assertTrue(bool(optimizer.state[model.weight]["direction"]["maximum"].gt(0).all()))

    def test_matrix_selection_and_invalid_inputs(self) -> None:
        """Check excluded matrices, dense-gradient requirements and parameter domains."""
        for factory in (ScheduleFreeAMSGradW, ScheduleFreeAMSGradMD):
            for options in ({"lr": -1.0}, {"warmup": -1}, {"betas": (0.9, 1.0)}, {"eps": 0.0}):
                with (
                    self.subTest(rule=factory.__name__, options=options),
                    self.assertRaises(ValueError),
                ):
                    factory([torch.nn.Parameter(torch.ones(2, 3))], **options)
            p = torch.nn.Parameter(torch.ones(2, 3))
            optimizer = factory([p])
            with self.assertRaises(ValueError):
                optimizer.step()
            with self.assertRaises(ValueError):
                optimizer.add_param_group({"params": [torch.nn.Parameter(torch.ones(1))]})
        for value in (0.0, math.inf, math.nan):
            with self.subTest(value=value), self.assertRaises(ValueError):
                ScheduleFreeAMSGradMD([torch.nn.Parameter(torch.full((2, 3), value))])
        excluded = torch.nn.Parameter(torch.zeros(2, 3))
        optimizer = ScheduleFreeAMSGradMD([excluded], matrix_params=[])
        self.assertIn("maximum", optimizer.state[excluded])


if __name__ == "__main__":
    unittest.main()
