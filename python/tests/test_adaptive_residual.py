"""Independent source correspondence, conditional gain equations and state replay."""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from lab.dsl import AdaptiveResidual, CanonicalResidual, ResidualMaxAutotune, describe, swap
from lab.infrastructure.nn import build_model
from references.gpt_df1e9f7.gpt import GPT
from test_residual_transformer import recipe, train


def source(spec, seed=145):
    torch.manual_seed(seed)
    return GPT(11, spec.width, spec.context, spec.heads, spec.depth,
               use_checkpoint=False, pad_vocab_size_to=1)


def gated_equations(model, tokens, weights):
    """Independent recurrence over the frozen actual GPT's branch primitives."""
    cos, sin = model.rope_cos[:tokens.shape[1]], model.rope_sin[:tokens.shape[1]]
    x = model.embedding_norm(model.token_embedding_table(tokens))
    constant = (tokens != tokens[:, :1]).cumsum(1) == 0
    for index, block in enumerate(model.blocks):
        normalized = block.ln_1(x)
        branch, valid = block.sa(normalized, cos, sin, None, return_guard=True)
        constant = constant & valid[:, None]
        branch = block.attn_out_norm(branch)
        branch = (2 * (normalized @ weights[2 * index].T).sigmoid()) * branch
        branch = torch.where(constant[..., None], branch[:, :1], branch)
        x = x + block.residual_scale * branch
        normalized = block.ln_2(x)
        branch = block.ffn_out_norm(block.ffwd(normalized))
        branch = (2 * (normalized @ weights[2 * index + 1].T).sigmoid()) * branch
        branch = torch.where(constant[..., None], branch[:, :1], branch)
        x = x + block.residual_scale * branch
    x = torch.where(constant[..., None], x[:, :1], x)
    return model.lm_head(model.ln_f(x))


class AdaptiveResidualTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.tokens = torch.tensor([[1, 1, 1, 2], [3, 4, 5, 6]])

    def test_frozen_source_hashes_and_original_import_rewrite(self):
        root = Path(__file__).parent / "references/gpt_df1e9f7"
        manifest = json.loads((root / "manifest.json").read_text())
        self.assertTrue(manifest["commit"].startswith("df1e9f7"))
        for name, entry in manifest["files"].items():
            local = (root / name).read_bytes()
            self.assertEqual(hashlib.sha256(local).hexdigest(), entry["local_sha256"])
            original = local.replace(b"from .", b"from diff_gpt.domain.model.")
            self.assertEqual(hashlib.sha256(original).hexdigest(), entry["source_sha256"])

    def test_canonical_matches_frozen_actual_gpt_forward_and_every_gradient(self):
        spec = CanonicalResidual(8, 2, 2, 8)
        for clipped in (False, True):
            with self.subTest(clipped=clipped):
                expected, actual = source(spec).double(), build_model(spec, 11, 145).double()
                torch.testing.assert_close(actual.state_dict(), expected.state_dict(), rtol=0, atol=0)
                if clipped:
                    with torch.no_grad():
                        for model in (expected, actual):
                            model.blocks[0].sa.c_attn.weight.mul_(1e-14)
                a, b = actual(self.tokens), expected(self.tokens)[0]
                torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-14)
                a.square().mean().backward()
                b.square().mean().backward()
                reference = dict(expected.named_parameters())
                for name, value in actual.named_parameters():
                    torch.testing.assert_close(value.grad, reference[name].grad, rtol=1e-10, atol=1e-12, msg=name)

    def test_zero_gains_preserve_common_initialization_rng_forward_and_gradients(self):
        spec = CanonicalResidual(16, 3, 2, 8)
        base = build_model(spec, 11, 145)
        rng = torch.get_rng_state()
        candidate = build_model(AdaptiveResidual(16, 3, 2, 8), 11, 145)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertNotIn("mixing", describe(spec))
        with self.assertRaises(TypeError):
            CanonicalResidual(mixing="add")
        common = dict(base.named_parameters())
        for name, value in candidate.named_parameters():
            if name in common:
                torch.testing.assert_close(value, common[name], rtol=0, atol=0)
            else:
                self.assertIn("_gain.weight", name)
                self.assertTrue(torch.equal(value, torch.zeros_like(value)))
        extra = sum(p.numel() for p in candidate.parameters()) - sum(p.numel() for p in base.parameters())
        self.assertEqual(extra, 2 * spec.depth * spec.width)
        a, b = candidate(self.tokens), base(self.tokens)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        a.square().mean().backward()
        b.square().mean().backward()
        for name, value in candidate.named_parameters():
            if name in common:
                torch.testing.assert_close(value.grad, common[name].grad, rtol=0, atol=0, msg=name)
            else:
                self.assertGreater(value.grad.norm().item(), 0)

    def test_actual_measured_sizes_match_source_fp32_gradients_exactly(self):
        for prime in (97, 193):
            for seed in (11, 17):
                with self.subTest(prime=prime, seed=seed):
                    actual = build_model(CanonicalResidual(128, 4, 4, 33), prime, seed)
                    torch.manual_seed(seed)
                    expected = GPT(prime, 128, 33, 4, 4, use_checkpoint=False, pad_vocab_size_to=1)
                    torch.testing.assert_close(actual.state_dict(), expected.state_dict(), rtol=0, atol=0)
                    for tokens in (self.tokens, torch.tensor([[1, 2], [1, 1], [12, 33], [94, 96]])):
                        actual.zero_grad(set_to_none=True)
                        expected.zero_grad(set_to_none=True)
                        a, b = actual(tokens), expected(tokens)[0]
                        torch.testing.assert_close(a, b, rtol=0, atol=0)
                        a.square().mean().backward()
                        b.square().mean().backward()
                        reference = dict(expected.named_parameters())
                        for name, value in actual.named_parameters():
                            torch.testing.assert_close(value.grad, reference[name].grad, rtol=0, atol=0, msg=name)

    def test_nonzero_gains_match_independent_frozen_gpt_recurrence_and_gradients(self):
        spec = AdaptiveResidual(8, 2, 2, 8)
        actual, expected = build_model(spec, 11, 145).double(), source(spec).double()
        weights = []
        for block in actual.blocks:
            for gate in (block.attn_gain, block.ffn_gain):
                with torch.no_grad():
                    gate.weight.normal_(std=.2)
                weights.append(gate.weight.detach().clone().requires_grad_())
        a, b = actual(self.tokens), gated_equations(expected, self.tokens, weights)
        torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-14)
        a.square().mean().backward()
        b.square().mean().backward()
        reference = dict(expected.named_parameters())
        gate_index = 0
        for name, value in actual.named_parameters():
            if name in reference:
                gradient = reference[name].grad
            else:
                gradient = weights[gate_index].grad
                gate_index += 1
            torch.testing.assert_close(value.grad, gradient, rtol=1e-10, atol=1e-12, msg=name)

    def test_fullgraph_max_autotune_forward_backward(self):
        for device in ("cpu", "cuda"):
            if device == "cuda" and not torch.cuda.is_available():
                continue
            for kind in (CanonicalResidual, AdaptiveResidual):
                with self.subTest(device=device, kind=kind.__name__):
                    eager = build_model(kind(8, 2, 2, 8), 11, 145).to(device)
                    model = copy.deepcopy(eager)
                    compiled = torch.compile(model, fullgraph=True, mode="max-autotune")
                    inputs = self.tokens.to(device)
                    with sdpa_kernel(SDPBackend.MATH):
                        a, b = compiled(inputs), eager(inputs)
                        a.square().mean().backward()
                        b.square().mean().backward()
                    torch.testing.assert_close(a, b, rtol=2e-4, atol=2e-6)
                    for a, b in zip(model.parameters(), eager.parameters(), strict=True):
                        torch.testing.assert_close(a.grad, b.grad, rtol=5e-4, atol=2e-6)
                    torch._dynamo.reset()

    def test_compiled_complete_state_continuation_seals_final_test(self):
        spec = swap(recipe(), "model", AdaptiveResidual(8, 2, 2, 8))
        spec = swap(spec, "execution", ResidualMaxAutotune(device="cpu"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            _, straight = train(spec, path / "straight")
            train(swap(spec, "budget.updates", 4), path / "continued")
            _, continued = train(spec, path / "continued")
            a, b = straight.checkpoint("cpu"), continued.checkpoint("cpu")
            for key in ("model", "optimizer", "permutation"):
                torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
            for key in ("cursor", "rotary_random_state", "step", "examples_seen"):
                self.assertEqual(a[key], b[key])
            self.assertTrue(all("test" not in row for row in continued.records("history")))


if __name__ == "__main__":
    unittest.main()
