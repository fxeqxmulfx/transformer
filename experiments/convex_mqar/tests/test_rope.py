"""Mathematical RoPE and explicit-softmax references, gradients, and causality."""

import math
import unittest

import torch
from torch.nn import functional as F

from convex_mqar.rope import RopeTransformer, RotaryAttention

from .support import batch


def reference_rotation(x, base, offset=0):
    """Apply explicit 2x2 rotation matrices, separately at each position."""
    rotated = torch.empty_like(x)
    for position in range(x.shape[-2]):
        for r in range(x.shape[-1] // 2):
            theta = (position + offset) * base ** (-2 * r / x.shape[-1])
            matrix = x.new_tensor([[math.cos(theta), -math.sin(theta)],
                                   [math.sin(theta), math.cos(theta)]])
            rotated[..., position, 2*r:2*r+2] = x[..., position, 2*r:2*r+2] @ matrix.T
    return rotated


class RopeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)

    def test_rotation_matches_explicit_two_dimensional_matrices(self):
        for width, heads, base in ((8, 1, 10_000.), (12, 3, 97.)):
            attention = RotaryAttention(width, heads, base).double()
            x = torch.randn(2, heads, 7, width // heads, dtype=torch.float64)
            torch.testing.assert_close(attention.rotate(x), reference_rotation(x, base),
                                       rtol=1e-6, atol=1e-6)
            torch.testing.assert_close(attention.rotate(x)[..., 0, :], x[..., 0, :])
            torch.testing.assert_close(attention.rotate(x).square().sum(-1), x.square().sum(-1))

    def test_rotated_dot_products_depend_on_relative_position(self):
        q, k = torch.randn(1, 1, 5, 8, dtype=torch.float64), torch.randn(1, 1, 5, 8,
                                                                                  dtype=torch.float64)
        attention = RotaryAttention(8, 1, 10_000.).double()
        actual = attention.rotate(q) @ attention.rotate(k).transpose(-1, -2)
        for i in range(5):
            for j in range(5):
                relative_k = reference_rotation(k[..., j:j+1, :], 10_000., j - i)
                expected = (q[..., i:i+1, :] * relative_k).sum(-1).squeeze(-1)
                torch.testing.assert_close(actual[..., i, j], expected, atol=1e-6, rtol=1e-6)
        shifted_q = reference_rotation(q, 10_000., 31)
        shifted_k = reference_rotation(k, 10_000., 31)
        torch.testing.assert_close(shifted_q @ shifted_k.transpose(-1, -2), actual,
                                   atol=1e-6, rtol=1e-6)

    def test_attention_matches_explicit_causal_softmax_for_multiple_heads(self):
        for width, heads in ((8, 1), (12, 3)):
            attention = RotaryAttention(width, heads, 10_000.).double()
            x = torch.randn(2, 7, width, dtype=torch.float64)
            projected = F.linear(x, attention.qkv.weight)
            split = projected.split(width, dim=-1)
            q, k, v = [p.reshape(2, 7, heads, width // heads).transpose(1, 2) for p in split]
            q, k = reference_rotation(q, 10_000.), reference_rotation(k, 10_000.)
            scores = q @ k.transpose(-1, -2) / math.sqrt(width // heads)
            future = torch.triu(torch.ones(7, 7, dtype=torch.bool), diagonal=1)
            probabilities = scores.masked_fill(future, -torch.inf).softmax(-1)
            mixed = (probabilities @ v).transpose(1, 2).reshape(2, 7, width)
            expected = F.linear(mixed, attention.output.weight)
            torch.testing.assert_close(attention(x), expected, atol=1e-7, rtol=1e-6)

    def test_attention_derivatives_match_finite_differences(self):
        attention = RotaryAttention(4, 1, 100.).double()
        x = torch.randn(1, 3, 4, dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(attention, (x,), fast_mode=True))

    def test_query_selection_only_gathers_output_logits(self):
        model = RopeTransformer(64, 16, heads=2).eval()
        tokens, positions, _ = batch()
        positions = torch.cat((positions.flip(-1), positions[:, :1]), dim=1)
        with torch.no_grad():
            all_logits = model(tokens)
            expected = all_logits.gather(1, positions[..., None].expand(-1, -1, 64))
            torch.testing.assert_close(model(tokens, positions), expected)

    def test_transformer_causality_at_every_cut_and_batch_element(self):
        model = RopeTransformer(64, 16, heads=2).eval()
        tokens, _, _ = batch(count=2)
        with torch.no_grad():
            expected = model(tokens)
            for cut in (1, 4, 8, 15):
                changed = tokens.clone()
                changed[:, cut:] = (changed[:, cut:] + 13) % 64
                torch.testing.assert_close(model(changed)[:, :cut], expected[:, :cut])
                torch.testing.assert_close(model(tokens[:, :cut]), expected[:, :cut])

    def test_future_inputs_have_zero_gradient_to_past_attention_outputs(self):
        attention = RotaryAttention(8, 2, 10_000.)
        x = torch.randn(2, 7, 8, requires_grad=True)
        attention(x)[:, 3].square().sum().backward()
        self.assertEqual(x.grad[:, 4:].count_nonzero().item(), 0)
        self.assertGreater(x.grad[:, :4].abs().sum().item(), 0)

    def test_all_trainable_parameters_receive_finite_nonzero_gradients(self):
        model = RopeTransformer(64, 16, heads=2)
        tokens, positions, labels = batch()
        F.cross_entropy(model(tokens, positions).flatten(0, 1), labels.flatten()).backward()
        for name, parameter in model.named_parameters():
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertGreater(parameter.grad.abs().sum().item(), 0)

    def test_bf16_logits_and_backpropagation_remain_finite_on_cpu(self):
        model = RopeTransformer(64, 16)
        tokens, positions, labels = batch()
        reference = model(tokens, positions).detach()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            logits = model(tokens, positions)
            loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten())
        torch.testing.assert_close(logits.float(), reference, atol=0.005, rtol=0.03)
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in model.parameters()))

    def test_invalid_head_geometry_is_rejected(self):
        for width, heads in ((7, 1), (10, 3), (12, 4)):
            with self.subTest(width=width, heads=heads), self.assertRaises(ValueError):
                RotaryAttention(width, heads, 10_000.)
