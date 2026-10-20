"""The derived YaRN block composes with the lab's scores and model builders."""

import math
import unittest

import torch

from lab.dsl import (FusedQKV, LearnedScaledDot, PerHeadQKV, QKNorm, ScaledDot, Softmax, Sparsemax, YaRN,
                     describe, extend_context, fingerprint, swap)
from lab.infrastructure.nn import build_model
from lab.infrastructure.nn.attention import scores_module, softmax
from lab.infrastructure.nn.yarn import retention_logits

from examples import reference, rope


def yarn_model(*, scores=ScaledDot(), projections=FusedQKV(), fused=True, interleaved=True):
    spec = rope(width=16, depth=1, heads=2, context=9)
    spec = swap(spec, "block.attention.scores", scores)
    spec = swap(spec, "block.attention.projections", projections)
    spec = swap(spec, "block.attention.weights", Softmax(fused=fused))
    return swap(spec, "positions", YaRN(original_context=3, interleaved=interleaved))


class YaRNModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_dsl_records_original_context_without_tunable_gain(self):
        self.assertEqual(describe(YaRN(4096)),
                         {"type": "YaRN", "original_context": 4096, "theta": 10_000., "interleaved": False})
        self.assertNotEqual(fingerprint(yarn_model()), fingerprint(swap(yarn_model(), "positions.original_context", 4)))
        for original in (0, 1, True, 3., math.inf):
            with self.subTest(original=original), self.assertRaisesRegex(ValueError, "original_context"):
                YaRN(original)
        for theta in (1, math.inf, math.nan, True):
            with self.subTest(theta=theta), self.assertRaisesRegex(ValueError, "base"):
                YaRN(3, theta=theta)
        with self.assertRaisesRegex(ValueError, "interleaved"):
            YaRN(3, interleaved=1)
        with self.assertRaisesRegex(ValueError, "Softmax"):
            swap(yarn_model(), "block.attention.weights", Sparsemax())
        with self.assertRaisesRegex(ValueError, "even head width"):
            swap(yarn_model(), "width", 6)

    def test_extension_copies_metadata_and_keeps_the_original_across_repeated_extensions(self):
        base = swap(rope(width=16, context=3), "positions.theta", 500_000.)
        extended = extend_context(base, 9)
        self.assertEqual(extended.context, 9)
        self.assertEqual(extended.positions, YaRN(3, theta=500_000., interleaved=True))
        self.assertEqual(base.context, 3)
        again = extend_context(extended, 27)
        self.assertEqual(again.context, 27)
        self.assertEqual(again.positions, extended.positions)

    def test_extension_refuses_missing_rotary_metadata_and_invalid_targets(self):
        with self.assertRaisesRegex(TypeError, "RoPE or YaRN"):
            extend_context(reference(), 100)
        for target in (2, 9., True):
            with self.subTest(target=target), self.assertRaisesRegex(ValueError, "Extended context"):
                extend_context(rope(context=3), target)

    def test_native_context_matches_rope_outputs_and_gradients_bit_for_bit(self):
        for scores in (ScaledDot(), QKNorm()):
            spec = swap(rope(width=16, depth=1, heads=2, context=9), "block.attention.scores", scores)
            native = build_model(spec, 13, seed=1)
            derived = build_model(swap(spec, "positions", YaRN(9, interleaved=True)), 13, seed=1)
            tokens = torch.tensor([[0, 1, 2, 3, 4, 5]])
            outputs = [net(tokens) for net in (native, derived)]
            self.assertTrue(torch.equal(*outputs))
            for out in outputs:
                out.square().sum().backward()
            for (na, a), (nb, b) in zip(native.named_parameters(), derived.named_parameters()):
                self.assertEqual(na, nb)
                self.assertTrue(torch.equal(a.grad, b.grad))

    def test_checkpoint_keys_and_parameters_survive_context_extension(self):
        base = swap(rope(width=16, depth=1, heads=2, context=3), "block.attention.scores", QKNorm())
        native = build_model(base, 13, seed=1)
        spec = swap(swap(base, "positions", YaRN(3, interleaved=True)), "context", 9)
        derived = build_model(spec, 13, seed=2)
        derived.load_state_dict(native.state_dict(), strict=True)
        self.assertEqual(list(native.state_dict()), list(derived.state_dict()))
        for a, b in zip(native.parameters(), derived.parameters()):
            self.assertTrue(torch.equal(a, b))

    def test_all_score_and_projection_choices_match_independent_row_calculation(self):
        for score_spec in (ScaledDot(), QKNorm(initial_scale=2.), LearnedScaledDot(initial_scale=2.)):
            for projections in (FusedQKV(), PerHeadQKV()):
                for interleaved in (False, True):
                    with self.subTest(scores=score_spec, projections=projections, interleaved=interleaved):
                        spec = yarn_model(scores=score_spec, projections=projections, interleaved=interleaved)
                        net = build_model(spec, 13, seed=7).double()
                        attention = net.blocks[0].attention
                        self.assertFalse(attention.fused)
                        raw = scores_module(score_spec, 2, 8).double()
                        raw.load_state_dict(attention.scores.state_dict())
                        x = torch.randn(2, 9, 16, dtype=torch.float64)
                        rotary = (net.cos, net.sin, interleaved)
                        groups = []
                        for q, k, v, index in attention.projections(x):
                            unscaled = raw(q, k, rotary, index)
                            expected = torch.zeros_like(v)
                            for row in range(9):
                                visible = unscaled[..., row, :row + 1]
                                if row >= 3:
                                    values = visible.topk(2, dim=-1).values
                                    gap = values[..., 0] - values[..., 1]
                                    gain = 1 + math.log(row / 2) / gap
                                    visible = visible * gain[..., None]
                                expected[..., row, :] = (torch.softmax(visible, dim=-1)[..., None, :]
                                                         @ v[..., :row + 1, :]).squeeze(-2)
                            groups.append(expected)
                            direct = attention.scores.attend(q, k, v, rotary, index)
                            torch.testing.assert_close(direct, expected)
                        expected = attention.output(attention.projections.merge(groups))
                        torch.testing.assert_close(attention(x, rotary), expected)

    def test_fused_request_has_the_same_extended_outputs_and_gradients(self):
        a = build_model(yarn_model(fused=True), 13, seed=4)
        b = build_model(yarn_model(fused=False), 13, seed=4)
        tokens = torch.arange(9)[None]
        outputs = [net(tokens) for net in (a, b)]
        self.assertTrue(torch.equal(*outputs))
        for out in outputs:
            out.square().sum().backward()
        for pa, pb in zip(a.parameters(), b.parameters()):
            self.assertTrue(torch.equal(pa.grad, pb.grad))

    def test_extended_model_is_causal_and_trains_with_finite_gradients(self):
        net = build_model(swap(yarn_model(scores=QKNorm()), "depth", 2), 13, seed=2)
        tokens = torch.arange(9)[None]
        full = net(tokens)
        for length in (1, 3, 5):
            torch.testing.assert_close(net(tokens[:, :length]), full[:, :length], atol=1e-7, rtol=1e-5)
        loss = torch.nn.functional.cross_entropy(full[:, :-1].flatten(0, 1), tokens[:, 1:].flatten())
        loss.backward()
        self.assertTrue(all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                            for parameter in net.parameters()))
        torch.optim.AdamW(net.parameters(), lr=1e-3).step()
        self.assertTrue(torch.isfinite(net(tokens)).all())

    def test_half_precision_model_outputs_and_gradients_stay_finite(self):
        for dtype in (torch.float16, torch.bfloat16):
            net = build_model(yarn_model(scores=QKNorm()), 13, seed=2).to(dtype)
            outputs = net(torch.arange(9)[None])
            self.assertEqual(outputs.dtype, dtype)
            self.assertTrue(torch.isfinite(outputs).all())
            outputs.float().square().mean().backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in net.parameters()))

    def test_compiled_calibration_matches_eager_outputs_and_gradients(self):
        eager = (torch.arange(49, dtype=torch.float64).reshape(7, 7) / 3).requires_grad_()
        compiled = eager.detach().clone().requires_grad_()
        function = torch.compile(lambda x: softmax(retention_logits(x, 3)), backend="eager", fullgraph=True)
        a, b = softmax(retention_logits(eager, 3)), function(compiled)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        (a * torch.arange(7)).sum().backward()
        (b * torch.arange(7)).sum().backward()
        torch.testing.assert_close(eager.grad, compiled.grad, rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_cuda_model_forward_backward_and_cpu_probabilities(self):
        scores = torch.arange(49, dtype=torch.float32).reshape(7, 7) / 3
        torch.testing.assert_close(softmax(retention_logits(scores.cuda(), 3)).cpu(),
                                   softmax(retention_logits(scores, 3)))
        net = build_model(yarn_model(scores=QKNorm()), 13, seed=2).cuda()
        out = net(torch.arange(9, device="cuda")[None])
        self.assertTrue(torch.isfinite(out).all())
        out.square().mean().backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in net.parameters()))


if __name__ == "__main__":
    unittest.main()
