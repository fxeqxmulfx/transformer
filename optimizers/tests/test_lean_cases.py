"""Values checked from the actual definitions by scripts/OptimizerParity.lean.

The rational expected outputs below also appear literally in that script.
Neither test suite generates expected values from the PyTorch implementation.
"""

from __future__ import annotations

import unittest

import torch

import lean_optim as opt
from lean_optim.matrix import polar


class LeanCases(unittest.TestCase):
    """Numerical checks for lean."""

    def test_amsgrad_callback_and_native_adamw_have_different_bias_conventions(self) -> None:
        """Check amsgrad callback and native adamw have different bias conventions."""
        for cls, kwargs, expected in (
            (opt.AMSGrad, {}, 29 / 15),
            (opt.AdamW, {"weight_decay": 1 / 5}, 47 / 25),
        ):
            p = torch.nn.Parameter(torch.tensor([2.0], dtype=torch.float64))
            optimizer = cls([p], lr=1 / 10, betas=(1 / 2, 3 / 4), eps=1 / 2, **kwargs)
            p.grad = p.new_tensor([2.0])
            optimizer.step()
            self.assertAlmostEqual(float(p.detach()), expected, places=14)

    def test_adamx_rescales_the_existing_maximum_before_comparison(self) -> None:
        """Check adamx rescales the existing maximum before comparison."""
        p = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))
        optimizer = opt.AdamX([p], betas=(1 / 2, 0.0), beta1_decay=opt.Inverse())
        optimizer.state[p].update(
            m=torch.zeros_like(p),
            v=torch.zeros_like(p),
            maximum=p.new_tensor([4.0]),
            step=p.new_tensor(1.0),
        )
        p.grad = p.new_tensor([1.0])
        optimizer.step()
        self.assertEqual(float(optimizer.state[p]["maximum"]), 9.0)

    def test_adagrad_and_adamnc_mean_square_at_update_two(self) -> None:
        """Check adagrad and adamnc mean square at update two."""
        for cls in (opt.AdaGrad, opt.AdamNC):
            p = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))
            optimizer = cls([p])
            optimizer.state[p].update(
                m=torch.zeros_like(p), v=p.new_tensor([4.0]), step=p.new_tensor(1.0)
            )
            p.grad = p.new_tensor([4.0])
            optimizer.step()
            self.assertEqual(float(optimizer.state[p]["v"]), 10.0)

    def test_muon_unit_polynomial_and_magma_zero_alignment(self) -> None:
        """Check muon unit polynomial and magma zero alignment."""
        self.assertAlmostEqual(
            float(polar(torch.ones(1, 1, dtype=torch.float64), steps=1)), 701 / 1000, places=14
        )
        p = torch.nn.Parameter(torch.ones(1, 1, dtype=torch.float64))
        optimizer = opt.Magma(opt.SGD([p]), updates=1, survival=1.0)
        p.grad = torch.zeros_like(p)
        optimizer.step()
        self.assertEqual(optimizer.report()["magma"]["blocks"]["0"]["final_damping"], 1 / 2)

    def test_schedule_free_explicit_evaluation_weights(self) -> None:
        """Check schedule free explicit evaluation weights."""
        p = torch.nn.Parameter(torch.tensor([2.0], dtype=torch.float64))
        optimizer = opt.ScheduleFreeAdamW(
            [p], lr=1 / 5, betas=(3 / 5, 0.0), weight_decay=3 / 10, eps=1 / 2
        )
        p.grad = p.new_tensor([2.0])
        optimizer.step()
        optimizer.eval()
        self.assertAlmostEqual(float(p.detach()), 43 / 25, places=14)
        q = torch.nn.Parameter(torch.tensor([2.0], dtype=torch.float64))
        sgd = opt.ScheduleFreeSGD([q], lr=1.0)
        q.grad = q.new_tensor([1.0])
        sgd.step()
        sgd.eval()
        self.assertEqual(float(q.detach()), 3 / 2)

    def test_fresh_fisher_factor_weight_and_rejected_descent_candidate(self) -> None:
        """Check fresh fisher factor weight and rejected descent candidate."""
        model = torch.nn.Linear(1, 1, bias=False, dtype=torch.float64)
        optimizer = opt.AdaFisher(model, gamma=4 / 5)
        p = model.weight
        optimizer.state[p].update(
            h=p.new_tensor([2.0]),
            s=p.new_tensor([2.0]),
            m=torch.zeros_like(p),
            step=p.new_tensor(0.0),
        )
        optimizer.factors.fresh[p] = (p.new_tensor([7.0]), p.new_tensor([7.0]))
        p.grad = torch.zeros_like(p)
        optimizer.step()
        self.assertAlmostEqual(float(optimizer.state[p]["h"]), 6.0)
        optimizer.close()
        q = torch.nn.Parameter(torch.tensor([-3.0], dtype=torch.float64))
        guarded = opt.Guarded(opt.SGD([q], lr=0.1, weight_decay=1.0), sigma=0.5)
        q.grad = q.new_tensor([2.0])
        guarded.step()
        self.assertEqual(guarded.report()["guard"]["accepted"], 0)
        self.assertAlmostEqual(float(q.detach()), -3.2)


if __name__ == "__main__":
    unittest.main()
