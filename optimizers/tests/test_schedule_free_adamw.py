"""Revised AdamW against published 1.4.1, Lean checkpoints and mode lifecycles."""

from __future__ import annotations

import copy
import math
import unittest
from itertools import product
from typing import get_type_hints

import torch
from _schedule_free_upstream import upstream_adamw

from lean_optim import ScheduleFreeAdamW, ScheduleFreeAdamWPaper


class ScheduleFreeAdamWTests(unittest.TestCase):
    """Check the authors' recurrence without deriving expectations from this port."""

    def test_published_scalar_and_foreach_trajectories(self) -> None:
        """Compare warmup, changing rates, decay, signed gradients and both modes."""
        devices = ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",)
        cases = product(
            devices,
            (torch.float32, torch.float64),
            (0, 4),
            (0.0, 0.999),
            (0.0, 0.125),
            (False, True),
        )
        for device, dtype, warmup, beta2, decay, foreach in cases:
            with self.subTest(
                device=device, dtype=dtype, warmup=warmup, beta2=beta2, decay=decay, foreach=foreach
            ):
                values = torch.arange(1, 7, device=device, dtype=dtype).reshape(2, 3).T / 8
                parameters = [
                    torch.nn.Parameter(values.clone()),
                    torch.nn.Parameter(values[:, 0].clone()),
                ]
                reference = [torch.nn.Parameter(p.detach().clone()) for p in parameters]
                options = {"lr": 0.125, "betas": (0.6, beta2), "eps": 0.05, "weight_decay": decay}
                tolerance = 1e-12 if dtype == torch.float64 else 5e-6
                local = ScheduleFreeAdamW(parameters, warmup=warmup, **options)
                upstream = upstream_adamw()(
                    reference, warmup_steps=warmup, foreach=foreach, **options
                )
                upstream.train()
                for index, rate in enumerate((0.0, 0.125, 0.0625, 0.0, 0.0, 0.25, 0.01, 0.0)):
                    local.param_groups[0]["lr"] = upstream.param_groups[0]["lr"] = rate
                    for p, q in zip(parameters, reference, strict=True):
                        p.grad = torch.full_like(p, (-1) ** index * (index + 1) / 8)
                        q.grad = p.grad.clone()
                    saved = [p.grad.clone() for p in parameters]
                    local.step()
                    upstream.step()
                    for p, q, gradient in zip(parameters, reference, saved, strict=True):
                        torch.testing.assert_close(p.grad, gradient, rtol=0, atol=0)
                        torch.testing.assert_close(p, q, rtol=tolerance, atol=tolerance)
                        for key, upstream_key in (("base", "z"), ("second", "exp_avg_sq")):
                            torch.testing.assert_close(
                                local.state[p][key],
                                upstream.state[q][upstream_key],
                                rtol=tolerance,
                                atol=tolerance,
                            )
                    self.assertEqual(
                        local.param_groups[0]["maximum"], upstream.param_groups[0]["lr_max"]
                    )
                    self.assertEqual(
                        local.param_groups[0]["total"], upstream.param_groups[0]["weight_sum"]
                    )
                    for _ in range(2):
                        local.eval()
                        upstream.eval()
                        for p, q in zip(parameters, reference, strict=True):
                            torch.testing.assert_close(p, q, rtol=tolerance, atol=tolerance)
                    for _ in range(2):
                        local.train()
                        upstream.train()

    def test_lean_warmup_and_bias_corrected_variance_checkpoints(self) -> None:
        """Use the same rational states as scripts/ScheduleFreeAdamWParity.lean."""
        p = torch.nn.Parameter(torch.tensor([2.0], dtype=torch.float64))
        optimizer = ScheduleFreeAdamW(
            [p], lr=1 / 2, betas=(1 / 2, 1 / 2), weight_decay=1 / 8, eps=1.0, warmup=2
        )
        rows = (
            (1, 29 / 16, 29 / 16, 1 / 2, 29 / 16, 1 / 16, 1 / 4),
            (11, 15559 / 12800, 1471 / 1280, 243 / 4, 2051 / 1600, 5 / 16, 1 / 2),
            (9, 2971973 / 3686400, 127641 / 204800, 567 / 8, 455801 / 460800, 9 / 16, 1 / 2),
        )
        for gradient, query, base, second, evaluation, total, maximum in rows:
            p.grad = p.new_tensor([gradient])
            optimizer.step()
            self.assertAlmostEqual(p.item(), query, places=14)
            self.assertAlmostEqual(optimizer.state[p]["base"].item(), base, places=14)
            self.assertAlmostEqual(optimizer.state[p]["second"].item(), second, places=14)
            self.assertEqual(optimizer.param_groups[0]["total"], total)
            self.assertEqual(optimizer.param_groups[0]["maximum"], maximum)
            optimizer.eval()
            self.assertAlmostEqual(p.item(), evaluation, places=14)
            optimizer.train()

    def test_falling_and_zero_rates_keep_maximum_weighted_averaging(self) -> None:
        """At zero rate z freezes, while x moves and both clocks advance."""
        p = torch.nn.Parameter(torch.tensor([2.0], dtype=torch.float64))
        optimizer = ScheduleFreeAdamW([p], betas=(1 / 2, 0.0), weight_decay=1 / 8, eps=1.0)
        for rate, query, base, evaluation in (
            (1 / 2, 13 / 8, 13 / 8, 13 / 8),
            (1 / 4, 1529 / 1024, 371 / 256, 787 / 512),
            (0.0, 757 / 512, 371 / 256, 193 / 128),
        ):
            optimizer.param_groups[0]["lr"] = rate
            p.grad = torch.ones_like(p)
            optimizer.step()
            self.assertAlmostEqual(p.item(), query, places=14)
            self.assertAlmostEqual(optimizer.state[p]["base"].item(), base, places=14)
            optimizer.eval()
            self.assertAlmostEqual(p.item(), evaluation, places=14)
            optimizer.train()
        self.assertEqual(optimizer.param_groups[0]["step"], 3)
        self.assertEqual(optimizer.param_groups[0]["total"], 3 / 4)
        self.assertEqual(set(optimizer.state[p]), {"base", "second"})

    def test_checkpoint_resumes_from_both_modes_and_rejects_paper_histories(self) -> None:
        """Persist group clocks and parameter mode without serializing workspace."""
        for training in (False, True):
            with self.subTest(training=training):
                p = torch.nn.Parameter(torch.tensor([2.0], dtype=torch.float64))
                optimizer = ScheduleFreeAdamW([p], lr=0.125)
                for _ in range(3):
                    p.grad = torch.ones_like(p)
                    optimizer.step()
                if not training:
                    optimizer.eval()
                q = torch.nn.Parameter(p.detach().clone())
                restored = ScheduleFreeAdamW([q], lr=0.9)
                restored.load_state_dict(copy.deepcopy(optimizer.state_dict()))
                self.assertEqual(restored.param_groups[0]["training"], training)
                self.assertFalse(restored.workspace.buffers)
                for _ in range(3):
                    for solver, parameter in ((optimizer, p), (restored, q)):
                        solver.train()
                        parameter.grad = torch.full_like(parameter, -0.25)
                        solver.step()
                    torch.testing.assert_close(p, q, rtol=0, atol=0)
                    for key in ("base", "second"):
                        torch.testing.assert_close(
                            optimizer.state[p][key], restored.state[q][key], rtol=0, atol=0
                        )
                paper = ScheduleFreeAdamWPaper([q])
                with self.assertRaisesRegex(ValueError, "ScheduleFreeAdamWPaper"):
                    restored.load_state_dict(paper.state_dict())

    def test_validation_precedes_history_changes(self) -> None:
        """Reject unsupported hyperparameters, layouts and absent gradients."""
        for options in (
            {"betas": (0.0, 0.999)},
            {"betas": (1.0, 0.999)},
            {"betas": (0.9, 1.0)},
            {"lr": math.inf},
            {"eps": 0.0},
            {"weight_decay": math.nan},
            {"warmup": True},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                ScheduleFreeAdamW(torch.nn.Linear(2, 2), **options)
        for layout in (torch.sparse_coo, torch.sparse_csr):
            with self.subTest(layout=layout), self.assertRaisesRegex(ValueError, "dense real"):
                ScheduleFreeAdamW([torch.nn.Parameter(torch.eye(2).to_sparse(layout=layout))])
        p, q = [torch.nn.Parameter(torch.ones(2)) for _ in range(2)]
        optimizer = ScheduleFreeAdamW([p, q])
        p.grad = torch.ones_like(p)
        for gradient in (None, torch.ones_like(q).to_sparse()):
            q.grad = gradient
            with self.assertRaisesRegex(ValueError, "dense gradient"):
                optimizer.step()
            self.assertEqual(optimizer.param_groups[0]["step"], 0)
            self.assertEqual(optimizer.param_groups[0]["total"], 0)
            self.assertEqual(optimizer.state[p]["second"].count_nonzero().item(), 0)

    def test_tensor_closure_and_runtime_annotations(self) -> None:
        """Enable gradients for a closure and keep public type annotations resolvable."""
        p = torch.nn.Parameter(torch.ones(2))
        optimizer = ScheduleFreeAdamW([p])

        def closure() -> torch.Tensor:
            self.assertTrue(torch.is_grad_enabled())
            loss = p.square().sum()
            loss.backward()
            return loss

        loss = optimizer.step(closure)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.item(), 2.0)
        for method in (ScheduleFreeAdamW.__init__, ScheduleFreeAdamW.step):
            self.assertTrue(get_type_hints(method))
        optimizer.eval()
        with self.assertRaisesRegex(RuntimeError, "train"):
            optimizer.step()
