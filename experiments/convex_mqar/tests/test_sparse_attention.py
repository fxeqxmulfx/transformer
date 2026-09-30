"""Independent row QP, derivatives, and a single-change Transformer control."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.optimize import minimize
import torch
from torch.nn import functional as F

from convex_mqar import attention_ablation, benchmark, engine, kernels, sparse_attention
from convex_mqar.rope import RopeTransformer
from convex_mqar.sparse_attention import SparsemaxTransformer, causal_sparsemax

from .support import batch, config


class SparseAttentionTests(unittest.TestCase):
    def test_weights_match_independent_constrained_quadratic_optimizer(self):
        torch.manual_seed(9)
        scores = torch.randn(2, 5, 5, dtype=torch.float64)
        actual = causal_sparsemax(scores)
        for b in range(2):
            for i in range(5):
                z = scores[b, i, :i + 1].numpy()
                solved = minimize(lambda a: ((a - z) ** 2).sum() / 2,
                                  np.full(i + 1, 1 / (i + 1)),
                                  jac=lambda a: a - z, method="SLSQP",
                                  bounds=[(0, None)] * (i + 1),
                                  constraints={"type": "eq", "fun": lambda a: a.sum() - 1,
                                               "jac": lambda a: np.ones_like(a)},
                                  options={"ftol": 1e-12})
                self.assertTrue(solved.success)
                np.testing.assert_allclose(actual[b, i, :i + 1].numpy(), solved.x, atol=1e-8)
        self.assertEqual(actual.triu(1).count_nonzero().item(), 0)
        torch.testing.assert_close(actual.sum(-1), torch.ones(2, 5, dtype=torch.float64))
        self.assertTrue((actual >= 0).all())

    def test_projection_is_shift_invariant_and_support_is_sparse(self):
        scores = torch.tensor([[2., 99., 99.], [2., -3., 99.], [2., 1.6, -7.]],
                              dtype=torch.float64)
        expected = torch.tensor([[1., 0., 0.], [1., 0., 0.], [.7, .3, 0.]],
                                dtype=torch.float64)
        torch.testing.assert_close(causal_sparsemax(scores), expected)
        torch.testing.assert_close(causal_sparsemax(scores + 1e8), expected,
                                   atol=1e-8, rtol=1e-8)

    def test_projection_gradient_matches_finite_differences_and_excludes_future(self):
        torch.manual_seed(41)
        scores = torch.randn(2, 4, 4, dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(causal_sparsemax, (scores,), fast_mode=True))
        (causal_sparsemax(scores) * torch.randn_like(scores)).sum().backward()
        self.assertEqual(scores.grad.triu(1).count_nonzero().item(), 0)
        torch.testing.assert_close(scores.grad.sum(-1), torch.zeros(2, 4, dtype=torch.float64))

    def test_initial_weights_names_parameter_count_and_rng_are_identical(self):
        torch.manual_seed(29)
        ordinary = RopeTransformer(64, 16, heads=2)
        original_rng = torch.get_rng_state().clone()
        torch.manual_seed(29)
        sparse = SparsemaxTransformer(64, 16, heads=2)
        self.assertTrue(torch.equal(torch.get_rng_state(), original_rng))
        self.assertEqual(list(ordinary.state_dict()), list(sparse.state_dict()))
        for name, tensor in ordinary.state_dict().items():
            self.assertTrue(torch.equal(tensor, sparse.state_dict()[name]), name)
        self.assertEqual(sum(p.numel() for p in ordinary.parameters()),
                         sum(p.numel() for p in sparse.parameters()))

    def test_restoring_softmax_recovers_ordinary_model_outputs_and_gradients(self):
        torch.manual_seed(23)
        ordinary = RopeTransformer(64, 16, heads=2).double()
        sparse = SparsemaxTransformer(64, 16, heads=2).double()
        sparse.load_state_dict(ordinary.state_dict())
        tokens, positions, labels = batch(count=2)

        def softmax(scores):
            future = torch.ones_like(scores, dtype=torch.bool).triu(1)
            return scores.masked_fill(future, -torch.inf).softmax(-1)

        expected = ordinary(tokens, positions)
        F.cross_entropy(expected.flatten(0, 1), labels.flatten()).backward()
        with patch.object(sparse_attention, "causal_sparsemax", side_effect=softmax):
            actual = sparse(tokens, positions)
            F.cross_entropy(actual.flatten(0, 1), labels.flatten()).backward()
        torch.testing.assert_close(actual, expected, atol=1e-9, rtol=1e-7)
        for (_, p), (_, q) in zip(ordinary.named_parameters(), sparse.named_parameters()):
            torch.testing.assert_close(q.grad, p.grad, atol=1e-9, rtol=1e-7)

    def test_full_model_is_causal_and_every_parameter_receives_a_gradient(self):
        torch.manual_seed(7)
        model = SparsemaxTransformer(64, 16, heads=2)
        tokens, positions, labels = batch(count=2)
        expected = model(tokens)
        changed = tokens.clone()
        changed[:, 8:] = (changed[:, 8:] + 11) % 64
        torch.testing.assert_close(model(changed)[:, :8], expected[:, :8])
        torch.testing.assert_close(model(tokens[:, :8]), expected[:, :8])
        logits = model(tokens, positions)
        torch.testing.assert_close(logits, expected.gather(1, positions[..., None].expand(-1, -1, 64)))
        F.cross_entropy(logits.flatten(0, 1), labels.flatten()).backward()
        for name, p in model.named_parameters():
            self.assertIsNotNone(p.grad, name)
            self.assertTrue(torch.isfinite(p.grad).all(), name)
            self.assertGreater(p.grad.abs().sum().item(), 0, name)

    def test_bf16_projection_uses_fp32_and_full_model_has_finite_gradients(self):
        scores = torch.randn(2, 8, 8).bfloat16()
        weights = causal_sparsemax(scores)
        self.assertEqual(weights.dtype, torch.float32)
        torch.testing.assert_close(weights.sum(-1), torch.ones(2, 8))
        model, data = SparsemaxTransformer(64, 16), batch(count=2)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            loss = F.cross_entropy(model(*data[:2]).flatten(0, 1), data[2].flatten())
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in model.parameters()))

    def test_compiled_autograd_matches_eager_for_sparsemax(self):
        model, data, cfg = SparsemaxTransformer(16, 8), batch(count=2, length=8, vocab=16), config()
        expected = kernels.loss_for(model, cfg)(*data)
        expected.backward()
        gradients = {name: p.grad.clone() for name, p in model.named_parameters()}
        model.zero_grad(set_to_none=True)
        compiler = torch.compile
        with patch.object(kernels.torch, "compile", side_effect=lambda f, **kw:
                          compiler(f, backend="aot_eager", **kw)):
            actual = kernels.loss_for(model, cfg, compiled=True)(*data)
            actual.backward()
        torch.testing.assert_close(actual, expected)
        for name, p in model.named_parameters():
            torch.testing.assert_close(p.grad, gradients[name])

    def test_model_tags_prevent_loading_softmax_checkpoints_as_sparsemax(self):
        cfg, data = config(epochs=1), batch(count=2, length=8, vocab=16)
        with tempfile.TemporaryDirectory() as root, redirect_stdout(io.StringIO()):
            engine.train_run(cfg, 8, 8, .003, data, data, root, Path(root) / "status.json")
            with self.assertRaisesRegex(RuntimeError, "different model class"):
                engine.train_run(cfg, 8, 8, .003, data, data, root, Path(root) / "status.json",
                                 model_factory=SparsemaxTransformer)
            (Path(root) / "model.json").unlink()
            with self.assertRaisesRegex(RuntimeError, "untagged checkpoints"):
                engine.train_run(cfg, 8, 8, .003, data, data, root, Path(root) / "status.json",
                                 model_factory=SparsemaxTransformer)

    def test_real_pipeline_reuses_baseline_splits_and_never_retests_completed_runs(self):
        cfg = config(epochs=1, learning_rates=(.003,))
        with tempfile.TemporaryDirectory() as root, redirect_stdout(io.StringIO()):
            baseline, output, data = (Path(root) / name for name in ("baseline", "sparse", "data"))
            with patch.object(benchmark.torch, "set_num_threads"):
                original = benchmark.run(cfg, baseline, data)
            with patch.object(attention_ablation.torch, "set_num_threads"):
                report = attention_ablation.run(cfg, output, data, baseline)
            row = report["results"][0]
            self.assertEqual(row["splits"], original["results"][0]["splits"])
            self.assertEqual(row["softmax"], original["results"][0]["rope"])
            self.assertEqual(row["sparsemax_selected_run"]["parameters"],
                             row["softmax_selected_run"]["parameters"])
            with patch.object(attention_ablation.torch, "set_num_threads"), \
                    patch.object(attention_ablation, "train_run", side_effect=AssertionError("Do not train")), \
                    patch.object(attention_ablation, "evaluate", side_effect=AssertionError("Do not retest")):
                restored = attention_ablation.run(cfg, output, data, baseline)
            self.assertEqual(restored["results"], report["results"])
            self.assertEqual(json.loads((output / "status.json").read_text())["event"], "complete")


if __name__ == "__main__":
    unittest.main()
