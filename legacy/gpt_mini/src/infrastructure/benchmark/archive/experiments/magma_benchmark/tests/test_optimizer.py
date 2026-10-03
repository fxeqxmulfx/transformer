"""Analytic Magma contracts, written before the implementation and training."""

import copy
import math
import unittest

import torch

from experiments.gpt_mini import Config
from experiments.optimizer_benchmark.attention import make_model
from experiments.optimizer_benchmark.coordinate import CoordinateOptimizer
from experiments.optimizer_benchmark.matrix import MuonOptimizer
from experiments.magma_benchmark.optimizer import MagmaOptimizer, RMSPropOptimizer, cosine


NAME = "blocks.0.ffn.w_in.weight"


class MagmaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def scalar(self, rule="sgd", seed=0):
        p = torch.nn.Parameter(torch.tensor([[1.0]], dtype=torch.float64))
        base = CoordinateOptimizer([(NAME, p)], 0.1, rule=rule,
                                   decay=0.01 if rule == "adamw" else 0)
        return p, MagmaOptimizer(base, seed=seed)

    def test_current_dense_momentum_and_ema_use_algorithm_one_scaling(self):
        p, opt = self.scalar()
        opt.draw_masks = lambda count: [True] * count
        p.grad = torch.tensor([[2.0]], dtype=p.dtype)
        opt.step()
        scale = 0.45 + 0.1 / (1 + math.exp(-0.5))
        self.assertAlmostEqual(opt.state[p]["scale"].item(), scale)
        self.assertAlmostEqual(p.item(), 1 - 0.1 * scale * 2)
        self.assertAlmostEqual(opt.state[p]["alignment_momentum"].item(), 0.2)
        # New gradient opposes the previous EMA but agrees with the new EMA.
        p.grad.fill_(-20)
        opt.step()
        expected = 0.9 * scale + 0.1 / (1 + math.exp(-0.5))
        self.assertAlmostEqual(opt.state[p]["scale"].item(), expected)
        self.assertAlmostEqual(opt.state[p]["alignment_momentum"].item(), -1.82)

    def test_rejected_mask_keeps_weights_but_advances_moments_and_scale(self):
        p, opt = self.scalar("adamw")
        opt.draw_masks = lambda count: [False] * count
        p.grad = torch.tensor([[2.0]], dtype=p.dtype)
        opt.step()
        self.assertEqual(p.item(), 1)
        self.assertAlmostEqual(opt.base.state[p]["m"].item(), 0.2)
        self.assertAlmostEqual(opt.base.state[p]["v"].item(), 0.004)
        self.assertGreater(opt.state[p]["scale"].item(), 0.5)
        self.assertEqual((opt.steps, opt.base.steps), (1, 1))
        p.grad.fill_(-1)
        opt.step()
        self.assertAlmostEqual(opt.base.state[p]["m"].item(), 0.08)
        self.assertEqual(opt.diagnostics()["survival_fraction"], 0)

    def test_adamw_full_displacement_including_decay_matches_torch_reference(self):
        p, opt = self.scalar("adamw")
        ref = torch.nn.Parameter(p.detach().clone())
        adam = torch.optim.AdamW([ref], lr=0.1, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
        opt.draw_masks = lambda count: [True] * count
        moment, scale = 0.0, 0.5
        for gradient in (2.0, -1.0, 0.0):
            before = p.detach().clone()
            with torch.no_grad():
                ref.copy_(before)
            p.grad = torch.full_like(p, gradient)
            ref.grad = p.grad.clone()
            adam.step()
            moment = 0.9 * moment + 0.1 * gradient
            alignment = 0 if moment * gradient == 0 else math.copysign(1, moment * gradient)
            scale = 0.9 * scale + 0.1 / (1 + math.exp(-alignment / 2))
            opt.step()
            torch.testing.assert_close(p, before + scale * (ref.detach() - before), rtol=1e-12, atol=1e-12)

    def test_zero_cosine_and_positive_scale_invariant(self):
        zero, tiny = torch.zeros(2, dtype=torch.float64), torch.tensor([1e-100, -1e-100], dtype=torch.float64)
        self.assertEqual(cosine(zero, tiny).item(), 0)
        self.assertAlmostEqual(cosine(tiny, tiny).item(), 1)
        self.assertAlmostEqual(cosine(tiny, -tiny).item(), -1)
        p, opt = self.scalar()
        opt.draw_masks = lambda count: [True] * count
        for gradient in (0.0, 1.0, -0.1, -2.0, 0.0):
            p.grad = torch.full_like(p, gradient)
            opt.step()
            self.assertTrue(math.isfinite(p.item()))
            self.assertGreaterEqual(opt.state[p]["scale"].item(), 1 / (1 + math.exp(0.5)))
            self.assertLessEqual(opt.state[p]["scale"].item(), 1 / (1 + math.exp(-0.5)))

    def test_actual_hybrid_routes_unique_tied_weights_and_dense_auxiliaries(self):
        cfg = Config(vocab_size=65, n_layers=2, n_heads=4, d_model=128, d_ff=512, max_seq_len=64)
        model = make_model(cfg, "softmax", 0)
        base = MuonOptimizer(model.named_parameters(), 0.03)
        opt = MagmaOptimizer(base, seed=0)
        self.assertEqual((len(opt.masked_parameters), sum(p.numel() for p in opt.masked_parameters)), (8, 393216))
        self.assertEqual(sum(p.numel() for p in opt.param_groups[0]["params"]), 401544)
        self.assertIs(model.embed.weight, model.unembed.weight)
        initial = {p: p.detach().clone() for p in model.parameters()}
        opt.draw_masks = lambda count: [False] * count
        for p in model.parameters():
            p.grad = torch.full_like(p, 0.001)
        opt.step()
        for p in model.parameters():
            if p in opt.masked_parameters:
                torch.testing.assert_close(p, initial[p], rtol=0, atol=0)
                self.assertGreater(base.state[p]["momentum"].abs().sum().item(), 0)
            else:
                torch.testing.assert_close(p, initial[p] - 0.0015 * p.grad / (p.grad.abs() + 1e-8))
                self.assertGreater(base.state[p]["m"].abs().sum().item(), 0)

    def test_rmsprop_dense_analytic_step_and_unmasked_variance(self):
        p = torch.nn.Parameter(torch.tensor([[1.0]], dtype=torch.float64))
        base = RMSPropOptimizer([(NAME, p)], 0.001)
        p.grad = torch.full_like(p, 2)
        base.step()
        self.assertAlmostEqual(p.item(), 1 - 0.002 / (math.sqrt(0.004) + 1e-8))
        opt = MagmaOptimizer(base, seed=0)
        opt.draw_masks = lambda count: [False] * count
        old = p.detach().clone()
        p.grad.fill_(-1)
        opt.step()
        torch.testing.assert_close(p, old, rtol=0, atol=0)
        self.assertAlmostEqual(base.state[p]["v"].item(), 0.004996)
        self.assertAlmostEqual(base.state[p]["m"].item(), 0.08)

    def test_mask_rng_is_reproducible_independent_and_bernoulli(self):
        _, first = self.scalar(seed=3)
        _, second = self.scalar(seed=3)
        global_before = torch.random.get_rng_state().clone()
        masks = first.draw_masks(20000)
        self.assertEqual(masks, second.draw_masks(20000))
        torch.testing.assert_close(torch.random.get_rng_state(), global_before, rtol=0, atol=0)
        self.assertTrue(0.48 < sum(masks) / len(masks) < 0.52)
        pairs = zip(masks[::2], masks[1::2])
        self.assertTrue(0.23 < sum(a and b for a, b in pairs) / 10000 < 0.27)

    def test_checkpoint_restores_mask_rng_moments_steps_and_trajectory(self):
        p, opt = self.scalar("adamw", seed=4)
        for gradient in (2.0, -1.0, 0.5):
            p.grad = torch.full_like(p, gradient)
            opt.step()
        state = copy.deepcopy(opt.state_dict())
        other, restored = self.scalar("adamw", seed=99)
        with torch.no_grad():
            other.copy_(p)
        restored.load_state_dict(state)
        for gradient in (-3.0, 2.0, -0.7, 0.0):
            p.grad, other.grad = torch.full_like(p, gradient), torch.full_like(other, gradient)
            opt.step()
            restored.step()
            torch.testing.assert_close(other, p, rtol=0, atol=0)
        self.assertEqual(restored.diagnostics(), opt.diagnostics())

    def test_invalid_or_ambiguous_setup_is_rejected(self):
        p = torch.nn.Parameter(torch.ones(2, 2))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            CoordinateOptimizer([(NAME, p), (NAME, p)], 0.1)
        for tau in (0, -1, math.inf, math.nan):
            with self.subTest(tau=tau), self.assertRaises(ValueError):
                MagmaOptimizer(CoordinateOptimizer([(NAME, p)], 0.1, rule="sgd"), tau=tau)
        with self.assertRaises(ValueError):
            MagmaOptimizer(MuonOptimizer([(NAME, p)], 0.1, guarded=True))
