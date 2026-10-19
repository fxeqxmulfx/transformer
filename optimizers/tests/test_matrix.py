"""Check matrix recurrences against direct translations of the Lean equations."""

import copy
import math
import unittest

import torch

import lean_optim as opt
from lean_optim import roots


def inverse_fourth(matrix, eps):
    values, vectors = torch.linalg.eigh(matrix + eps * torch.eye(matrix.size(-1), dtype=matrix.dtype))
    return (vectors * values.pow(-0.25)) @ vectors.T


class MatrixTests(unittest.TestCase):
    def test_muon_nesterov_five_polynomial_steps_and_shape_scale(self):
        for shape in ((2, 3), (3, 2)):
            p = torch.nn.Parameter(torch.arange(6, dtype=torch.float64).reshape(shape) / 10)
            optimizer = opt.Muon([p], lr=0.2, momentum=0.6)
            moment, expected = torch.zeros_like(p), p.detach().clone()
            for sign in (1, -1, 0):
                g = sign * torch.arange(1, 7, dtype=torch.float64).reshape(shape)
                moment = 0.6 * moment + g
                x = 0.6 * moment + g
                x = x / x.norm() if x.norm() else torch.zeros_like(x)
                for _ in range(5):
                    gram = x @ x.T
                    x = 3.4445 * x - 4.775 * gram @ x + 2.0315 * gram @ gram @ x
                expected -= 0.2 * 0.2 * math.sqrt(max(shape)) * x
                p.grad = g
                optimizer.step()
                torch.testing.assert_close(p, expected, atol=3e-12, rtol=3e-12)
                torch.testing.assert_close(optimizer.state[p]["momentum"], moment)

    def test_dash_block_histories_grafting_and_zero_direction(self):
        p = torch.nn.Parameter(torch.arange(9, dtype=torch.float64).reshape(3, 3) / 10)
        optimizer = opt.Dash([p], lr=0.2, block=2, solver=opt.EVD(), beta=0.5, graft_betas=(0.5, 0.75), eps=0.25)
        g = torch.arange(-4, 5, dtype=torch.float64).reshape(3, 3)
        expected = p.detach().clone()
        for i in range(0, 3, 2):
            for j in range(0, 3, 2):
                block = g[i:i+2, j:j+2]
                left, right = 0.5 * block @ block.T, 0.5 * block.T @ block
                reference = 0.5 * block / ((0.25 * block.square()).sqrt() + 0.25)
                direction = inverse_fourth(left, 0.25) @ block @ inverse_fourth(right, 0.25)
                expected[i:i+2, j:j+2] -= 0.2 * direction * (reference.norm() / direction.norm())
        p.grad = g
        optimizer.step()
        torch.testing.assert_close(p, expected, atol=1e-13, rtol=1e-13)
        # Both preconditioned and graft reference directions vanish initially.
        q = torch.nn.Parameter(torch.ones(2, 2, dtype=torch.float64))
        zero = opt.Dash([q], solver=opt.EVD())
        q.grad = torch.zeros_like(q)
        zero.step()
        self.assertTrue(torch.equal(q, torch.ones_like(q)))

    def test_root_solvers_and_failed_power_iteration_start(self):
        # The ones start is orthogonal to the leading eigenvector of this SPD matrix.
        history = torch.tensor([[[2.0, -1.5], [-1.5, 2.0]]], dtype=torch.float64)
        scale = roots.guarded_scale(history)
        self.assertGreater(float(scale[0]), 3.5)
        exact = inverse_fourth(history[0], 0.5)
        for solver in (opt.EVD(), opt.NewtonDB(20), opt.CoupledNewton(30), opt.Chebyshev(80, 200)):
            with self.subTest(solver=solver):
                actual = roots.build_root(solver)(history, 0.5)[0]
                torch.testing.assert_close(actual, exact, atol=2e-13, rtol=2e-13)
        # Zero histories and the zero PI vector have a finite guarded scale.
        self.assertEqual(float(roots.guarded_scale(torch.zeros_like(history))[0]), 1.0)

    def test_adafisher_hooks_ema_new_factor_weight_and_corrected_momentum(self):
        model = torch.nn.Linear(2, 3, bias=False, dtype=torch.float64)
        optimizer = opt.AdaFisher(model, lr=0.1, beta=0.5, gamma=0.8, damping=0.25, weight_decay=0.2)
        inputs = torch.tensor([[1.0, 2.0], [3.0, -1.0]], dtype=torch.float64)
        coefficients = torch.tensor([[1.0, 2.0, 3.0], [2.0, -1.0, 0.5]], dtype=torch.float64)
        h, s, moment = torch.zeros(2, dtype=torch.float64), torch.zeros(3, dtype=torch.float64), torch.zeros_like(model.weight)
        for _ in range(2):
            optimizer.zero_grad()
            (model(inputs) * coefficients).sum().backward()
            t = _ + 1
            old = model.weight.detach().clone()
            h = 0.2 * h + 0.8 * inputs.square().sum(0)
            s = 0.2 * s + 0.8 * coefficients.square().sum(0)
            moment = 0.5 * moment + 0.5 * (coefficients.T @ inputs)
            hn, sn = (h - h.min()) / (h.max() - h.min()), (s - s.min()) / (s.max() - s.min())
            diagonal = sn[:, None] * hn[None, :] + 0.25
            expected = old - 0.1 * (moment / (1 - 0.5 ** t) / diagonal + 0.2 * old)
            optimizer.step()
            torch.testing.assert_close(model.weight, expected, atol=1e-13, rtol=1e-13)
            torch.testing.assert_close(optimizer.state[model.weight]["h"], h)
        optimizer.close()
        self.assertEqual(len(model._forward_hooks), 0)

    def test_magnitude_direction_sphere_and_checked_nonzero_fallback(self):
        p = torch.nn.Parameter(torch.tensor([[1.0, -1.0]], dtype=torch.float64))
        optimizer = opt.AMSGradMD([p], lr=0.1, betas=(0.5, 0.75), eps=0.25)
        p.grad = p.new_tensor([[2.0, -0.5]])
        original = p.detach().clone()
        gradient = p.grad.clone()
        raw = math.log(math.expm1(1.0))
        derivative = 1 - math.exp(-1)
        row_gradient = (original * gradient).sum(1) * derivative
        column_gradient = (original * gradient).sum(0) * derivative
        next_row = raw - 0.001 * 0.5 * row_gradient / (0.25 + 0.5 * row_gradient.abs())
        next_column = raw - 0.001 * 0.5 * column_gradient / (0.25 + 0.5 * column_gradient.abs())
        candidate = original - 0.1 * 0.5 * gradient / (0.25 + 0.5 * gradient.abs())
        candidate = candidate * original.norm() / candidate.norm()
        expected = candidate * torch.log1p(next_row.exp())[:, None] * torch.log1p(next_column.exp())[None, :]
        optimizer.step()
        torch.testing.assert_close(p, expected, atol=1e-13, rtol=1e-13)
        state = optimizer.state[p]
        recovered = p / (torch.nn.functional.softplus(state["row"])[:, None]
                         * torch.nn.functional.softplus(state["column"])[None, :])
        torch.testing.assert_close(recovered.norm(), state["radius"], atol=1e-13, rtol=1e-13)
        q = torch.nn.Parameter(torch.tensor([[1.0, -1.0]], dtype=torch.float64))
        guarded = opt.AMSGradMD([q], lr=0.1, guard_sigma=0.5)
        q.grad = q.detach().clone() / 0.1
        guarded.step()
        torch.testing.assert_close(q, q.new_tensor([[0.5, -0.5]]))
        self.assertEqual(guarded.report()["guard"]["halved"], 1)
        with self.assertRaises(ValueError):
            opt.AMSGradMD([q], guard_sigma=0.75)

    def test_magma_state_continues_with_identical_mask_plan(self):
        p = torch.nn.Parameter(torch.ones(2, 2, dtype=torch.float64))
        optimizer = opt.Magma(opt.SGD([p], lr=0.1), updates=5, seed=17)
        p.grad = torch.ones_like(p)
        optimizer.step()
        q = torch.nn.Parameter(p.detach().clone())
        continued = opt.Magma(opt.SGD([q], lr=0.1), updates=5, seed=17)
        continued.load_state_dict(copy.deepcopy(optimizer.state_dict()))
        for _ in range(4):
            p.grad, q.grad = torch.ones_like(p), torch.ones_like(q)
            optimizer.step()
            continued.step()
            self.assertTrue(torch.equal(p, q))
        self.assertEqual(optimizer.report(), continued.report())


if __name__ == "__main__":
    unittest.main()
