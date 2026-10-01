"""Check actual Lean recurrences, chain rules and the proved counterexample."""

import unittest

import torch
from torch.nn import functional as F

from gpt_mini.infrastructure.benchmark.optimizer_benchmark.coordinate import CoordinateOptimizer
from gpt_mini.infrastructure.benchmark.amsgrad_extensions_benchmark.optimizer import (
    ExtensionOptimizer, factor_gradients, inverse_softplus, md_proposal,
    new_md_state, checked_weights, rebalance,
)


class RuleTests(unittest.TestCase):
    def test_decay_is_absent_from_moments_and_uses_old_weight(self):
        p = torch.tensor([[2., -3.]], dtype=torch.float64)
        opt = ExtensionOptimizer({"weight": p}, "amsgradw", .05,
                                 beta=.6, beta2=.7, eps=.2, decay=.4)
        m, v, maximum = torch.zeros_like(p), torch.zeros_like(p), torch.zeros_like(p)
        expected = p.clone()
        for g in (torch.tensor([[1., -2.]]), torch.tensor([[-3., .2]]), torch.zeros_like(p)):
            g = g.to(p)
            m = .6 * m + .4 * g
            v = .7 * v + .3 * g.square()
            maximum = torch.maximum(maximum, v)
            expected = (1 - .05 * .4) * expected - .05 * m / (.2 + maximum.sqrt())
            opt.step({"weight": g})
            torch.testing.assert_close(p, expected, rtol=1e-13, atol=1e-13)
            for name, value in (("m", m), ("v", v), ("maximum", maximum)):
                torch.testing.assert_close(opt.state["weight"]["base"][name], value)

    def test_zero_decay_matches_existing_amsgrad(self):
        p = torch.tensor([[1., 2.]], dtype=torch.float64)
        q = torch.nn.Parameter(p.clone())
        opt = ExtensionOptimizer({"weight": p}, "amsgradw", .01, decay=0.)
        old = CoordinateOptimizer([("weight", q)], .01, rule="amsgrad")
        for g in (torch.ones_like(p), -torch.ones_like(p), .3 * torch.ones_like(p)):
            q.grad = g.clone()
            old.step()
            opt.step({"weight": g})
            torch.testing.assert_close(p, q, rtol=1e-13, atol=1e-13)
            for key in ("m", "v", "maximum"):
                torch.testing.assert_close(opt.state["weight"]["base"][key], old.state[q][key])

    def test_shifted_quadratic_converges_to_weighted_equilibrium(self):
        # AMSGradW.shiftedRun: eta1/4, epsilon1, decay2, beta9/10, beta2=1/2.
        p = torch.zeros(1, dtype=torch.float64)
        opt = ExtensionOptimizer({"weight": p}, "amsgradw", .25,
                                 beta=.9, beta2=.5, eps=1., decay=2.)
        for _ in range(1500):
            opt.step({"weight": p - 1})
        denominator = 1 + opt.state["weight"]["base"]["maximum"].sqrt()
        torch.testing.assert_close(p, 1 / (1 + 2 * denominator), rtol=0, atol=1e-12)
        self.assertGreater(abs((p - 1).item()), .5)

    def test_factor_gradients_match_reverse_ad(self):
        rng = torch.Generator().manual_seed(17)
        d = torch.randn(3, 4, generator=rng, dtype=torch.float64)
        row = torch.randn(3, generator=rng, dtype=torch.float64)
        col = torch.randn(4, generator=rng, dtype=torch.float64)
        g = torch.randn(3, 4, generator=rng, dtype=torch.float64)
        expected = torch.func.grad(lambda D, r, c:
            (D * F.softplus(r)[:, None] * F.softplus(c)[None, :] * g).sum(),
            argnums=(0, 1, 2))(d, row, col)
        actual = factor_gradients(d, row, col, g)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, rtol=1e-13, atol=1e-13)

    def test_full_proposal_matches_independent_autograd_factor_step(self):
        p = torch.tensor([[.5, -.2], [.3, .7]], dtype=torch.float64)
        state = new_md_state(p)
        memory = {name: {k: v.clone() for k, v in state[name].items()}
                  for name in ("direction", "row", "col")}
        raw_r, raw_c = state["raw_row"].clone(), state["raw_col"].clone()
        for g in (p.clone(), -p.clone(), p.clone() * .4):
            d = p / (F.softplus(raw_r)[:, None] * F.softplus(raw_c)[None, :])
            grads = torch.func.grad(lambda D, r, c:
                (D * F.softplus(r)[:, None] * F.softplus(c)[None, :] * g).sum(),
                argnums=(0, 1, 2))(d, raw_r, raw_c)
            updates = []
            for name, x, grad, lr in zip(("direction", "row", "col"),
                                       (d, raw_r, raw_c), grads, (.02, .03, .03)):
                s = memory[name]
                s["m"] = .6 * s["m"] + .4 * grad
                s["v"] = .7 * s["v"] + .3 * grad.square()
                s["maximum"] = torch.maximum(s["maximum"], s["v"])
                updates.append(x - lr * s["m"] / (.2 + s["maximum"].sqrt()))
            direction, raw_r, raw_c = updates
            direction = direction * state["radius"] / direction.norm()
            expected = direction * F.softplus(raw_r)[:, None] * F.softplus(raw_c)[None, :]
            actual, row, col = md_proposal(p, state, g, .02, .03, .6, .7, .2)
            torch.testing.assert_close(actual, expected, rtol=1e-13, atol=1e-13)
            torch.testing.assert_close(row, raw_r)
            torch.testing.assert_close(col, raw_c)
            for name in memory:
                for key in memory[name]:
                    torch.testing.assert_close(state[name][key], memory[name][key])
            state["raw_row"].copy_(row)
            state["raw_col"].copy_(col)
            p.copy_(actual)

    def test_lean_counterexample_and_guarded_repair(self):
        final = {}
        for method in ("amsgradmd", "amsgradmd_guarded"):
            p = torch.ones(1, 1, dtype=torch.float64)
            opt = ExtensionOptimizer({"blocks.0.weight": p}, method, .25,
                beta=0., beta2=0., eps=1., direction_rate=.25, gain_rate=.25,
                aux_rate=.25, sigma=.25)
            maxima = torch.zeros_like(p)
            for _ in range(400):
                opt.step({"blocks.0.weight": p + 1})
                state = opt.state["blocks.0.weight"]
                self.assertTrue(bool((state["direction"]["maximum"] >= maxima).all()))
                maxima = state["direction"]["maximum"].clone()
                d = p / (F.softplus(state["raw_row"])[:, None] * F.softplus(state["raw_col"])[None, :])
                torch.testing.assert_close(d.norm(), state["radius"], rtol=1e-12, atol=1e-12)
                if method == "amsgradmd":
                    self.assertGreater(p.item(), 0)
                    self.assertGreaterEqual((p + 1).abs().item(), 1)
            final[method] = p.item()
        self.assertGreater(final["amsgradmd"], 0)
        self.assertAlmostEqual(final["amsgradmd_guarded"], -1., places=9)

    def test_check_accepts_preserves_and_rejects_halves_zero_step(self):
        p = torch.ones(1, 1, dtype=torch.float64)
        g = torch.ones_like(p)
        weights, accepted, halved = checked_weights([p], [g], [.75 * p], [True], .25, .25)
        self.assertTrue(accepted.item())
        self.assertFalse(halved[0].item())
        torch.testing.assert_close(weights[0], .75 * p, rtol=0, atol=0)
        weights, accepted, halved = checked_weights([p], [4 * g], [0 * p], [True], .25, .25)
        self.assertFalse(accepted.item())
        self.assertTrue(halved[0].item())
        torch.testing.assert_close(weights[0], .5 * p, rtol=0, atol=0)

    def test_storage_repair_preserves_weight_columns_and_row_ratios(self):
        p = torch.tensor([[2., -1.], [3., 4.]], dtype=torch.float64)
        r = torch.tensor([-.2, .5], dtype=torch.float64)
        c = torch.tensor([.7, -.9], dtype=torch.float64)
        old_d = p / (F.softplus(r)[:, None] * F.softplus(c)[None, :])
        repaired = rebalance(p, r, c, torch.tensor(1.7, dtype=torch.float64))
        d = p / (F.softplus(repaired)[:, None] * F.softplus(c)[None, :])
        self.assertAlmostEqual(d.norm().item(), 1.7, places=12)
        torch.testing.assert_close(F.softplus(repaired)[0] / F.softplus(repaired)[1],
                                   F.softplus(r)[0] / F.softplus(r)[1])
        torch.testing.assert_close(F.softplus(repaired), old_d.norm() / 1.7 * F.softplus(r))

    def test_inverse_softplus_is_stable_without_clamping_gains(self):
        x = torch.tensor([1e-30, 1e-10, .5, 1., 100., 1000.], dtype=torch.float64)
        raw = inverse_softplus(x)
        self.assertTrue(raw.isfinite().all().item())
        torch.testing.assert_close(F.softplus(raw), x, rtol=1e-13, atol=0)

    def test_tied_embedding_is_auxiliary_and_buffers_are_distinct(self):
        hidden = torch.ones(3, 4)
        embedding = torch.ones(8, 4)
        temperature = torch.ones(2)
        opt = ExtensionOptimizer({"blocks.0.attn.qkv.weight": hidden,
            "embed.weight": embedding, "blocks.0.attn.log_alpha": temperature}, "amsgradmd", .001)
        self.assertEqual(opt.md_names, ("blocks.0.attn.qkv.weight",))
        self.assertEqual(opt.aux_names, ("embed.weight", "blocks.0.attn.log_alpha"))
        state = opt.state[opt.md_names[0]]
        pointers = [state[name][key].data_ptr() for name in ("direction", "row", "col")
                    for key in ("m", "v", "maximum")]
        self.assertEqual(len(set(pointers)), 9)
        with self.assertRaises(ValueError):
            ExtensionOptimizer({"embed.weight": embedding, "unembed.weight": embedding}, "amsgradmd", .001)

    def test_global_guard_certificate_includes_auxiliary_parameters(self):
        p = [torch.ones(2, 2, dtype=torch.float64), torch.ones(2, dtype=torch.float64)]
        g = [torch.tensor([[1., 2.], [-1., .5]], dtype=torch.float64), torch.tensor([3., -2.], dtype=torch.float64)]
        q = [-2 * p[0], 10 * p[1]]
        chosen, accepted, _ = checked_weights(p, g, q, [True, False], .3, .25)
        self.assertFalse(accepted.item())
        d = [(x - y) / .3 for x, y in zip(p, chosen)]
        norm_g = sum(x.square().sum() for x in g)
        norm_d = sum(x.square().sum() for x in d)
        alignment = sum((x * y).sum() for x, y in zip(g, d))
        self.assertGreaterEqual(alignment.item() + 1e-12, .25 * norm_g.item())
        self.assertLessEqual(norm_d.item(), norm_g.item() + 1e-12)
