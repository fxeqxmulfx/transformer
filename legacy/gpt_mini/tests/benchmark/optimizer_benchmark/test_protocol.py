import hashlib
import os
from pathlib import Path
import tempfile
import unittest

import torch
from torch.nn import functional as F

from gpt_mini.gpt_mini import Config
from gpt_mini.infrastructure.benchmark.optimizer_benchmark.attention import make_model, sparsemax
from gpt_mini.infrastructure.benchmark.optimizer_benchmark.data import TextData, evaluation_batches, select_learning_rates, training_starts, windows
from gpt_mini.infrastructure.benchmark.optimizer_benchmark.registry import METHODS, make_optimizer
from gpt_mini.infrastructure.benchmark.optimizer_benchmark.runner import evaluate


class AttentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_sparsemax_solves_simplex_projection(self):
        scores = torch.tensor([[0.7, 0.6, -1.0]], dtype=torch.float64, requires_grad=True)
        weights = sparsemax(scores)
        torch.testing.assert_close(weights, torch.tensor([[0.55, 0.45, 0.0]], dtype=scores.dtype))
        torch.testing.assert_close(weights.sum(-1), torch.ones(1, dtype=scores.dtype))
        self.assertTrue(bool((weights >= 0).all()))
        # KKT: equal active z_i-p_i and inactive scores below that threshold.
        threshold = (scores - weights)[weights > 0]
        torch.testing.assert_close(threshold, torch.full_like(threshold, 0.15))
        self.assertLess(scores[0, 2].item(), threshold[0].item())
        torch.testing.assert_close(sparsemax(scores + 10), weights)

    def test_sparsemax_gradient_matches_finite_differences(self):
        scores = torch.tensor([[0.7, 0.6, 0.5, -2.0]], dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(sparsemax, (scores,), eps=1e-6, atol=1e-5))
        (sparsemax(scores) * torch.tensor([[1.0, 2.0, 3.0, 100.0]])).sum().backward()
        torch.testing.assert_close(scores.grad, torch.tensor([[-1.0, 0.0, 1.0, 0.0]], dtype=scores.dtype))

    def test_sparsemax_large_translation_preserves_unit_mass(self):
        scores = torch.tensor([[0.5, 0.25, 0.0, float("-inf")]], dtype=torch.float32)
        expected = sparsemax(scores)
        for offset in (1000.0, 10000.0, 100000.0):
            with self.subTest(offset=offset):
                actual = sparsemax(scores + offset)
                torch.testing.assert_close(actual, expected, rtol=0, atol=1e-7)
                torch.testing.assert_close(actual.sum(-1), torch.ones(1), rtol=0, atol=1e-7)

    def test_sparsemax_nonfinite_scores_signal_numerical_failure(self):
        for invalid in (float("nan"), float("inf")):
            with self.subTest(score=invalid), self.assertRaises(FloatingPointError):
                sparsemax(torch.tensor([[invalid, 0.0]]))

    def test_causal_mask_and_nonempty_domain(self):
        scores = torch.tensor([[2.0, float("-inf"), float("-inf")],
                               [0.1, 0.2, float("-inf")], [0.7, 0.6, 0.5]])
        result = sparsemax(scores)
        torch.testing.assert_close(result.sum(-1), torch.ones(3))
        self.assertTrue(bool((result.triu(1) == 0).all()))
        with self.assertRaisesRegex(ValueError, "nonempty"):
            sparsemax(torch.full((1, 2), float("-inf")))

    def test_models_have_identical_initial_weights_and_tying(self):
        config = Config(vocab_size=8, n_layers=1, n_heads=2, d_model=16, d_ff=32, max_seq_len=8)
        models = [make_model(config, kind, 7) for kind in ["softmax", "sparsemax"]]
        for first, second in zip(models[0].parameters(), models[1].parameters()):
            torch.testing.assert_close(first, second, atol=0, rtol=0)
        for model in models:
            self.assertIs(model.embed.weight, model.unembed.weight)
            self.assertEqual(len(list(model.parameters())), len({id(p) for p in model.parameters()}))
            self.assertEqual(sum(id(p) == id(model.embed.weight) for p in model.parameters()), 1)

    def test_actual_models_ignore_future_tokens(self):
        config = Config(vocab_size=8, n_layers=2, n_heads=2, d_model=16, d_ff=32, max_seq_len=8)
        first = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]])
        changed = torch.tensor([[0, 1, 2, 3, 7, 7, 7, 7]])
        for kind in ["softmax", "sparsemax"]:
            with self.subTest(attention=kind), torch.no_grad():
                model = make_model(config, kind, 7).eval()
                torch.testing.assert_close(model(first)[:, :4], model(changed)[:, :4], rtol=0, atol=1e-6)


class ProtocolTests(unittest.TestCase):
    def test_nonfinite_evaluation_is_rejected_before_persistence(self):
        class InvalidModel(torch.nn.Module):
            def forward(self, tokens):
                return torch.full((*tokens.shape, 2), float("nan"))

        with self.assertRaisesRegex(FloatingPointError, "held-out"):
            evaluate(InvalidModel(), torch.zeros(8, dtype=torch.long), 2, 2)

    def test_disjoint_splits_and_next_token_boundaries(self):
        text = "abcdefghij" * 100
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "text.txt"
            path.write_text(text)
            data = TextData.load(path)
        self.assertEqual(data.boundaries, (900, 950, 1000))
        self.assertEqual([len(data.train), len(data.validation), len(data.test)], [900, 50, 50])
        self.assertNotEqual(data.train.data_ptr(), data.validation.data_ptr())
        self.assertEqual(data.sha256, hashlib.sha256(text.encode()).hexdigest())
        starts = training_starts(len(data.train), 8, 4, 20, 0)
        self.assertTrue(bool((starts + 8 < len(data.train)).all()))
        x, y = windows(data.train, starts[0], 8)
        torch.testing.assert_close(x[:, 1:], y[:, :-1])
        batches = list(evaluation_batches(data.test, 8, 4))
        self.assertEqual(sum(y.numel() for _, y in batches), 48)

    def test_frozen_minibatch_plan(self):
        first = training_starts(10000, 64, 32, 20, 0)
        second = training_starts(10000, 64, 32, 20, 0)
        torch.testing.assert_close(first, second, atol=0, rtol=0)
        self.assertFalse(torch.equal(first, training_starts(10000, 64, 32, 20, 1)))

    def test_test_loss_cannot_select_a_learning_rate(self):
        rows = [dict(phase="screen", method="amsgrad", status="ok", lr=0.001,
                     validation_loss=2.0, test_loss=100),
                dict(phase="screen", method="amsgrad", status="ok", lr=0.003,
                     validation_loss=3.0, test_loss=0.1),
                dict(phase="final", method="amsgrad", status="ok", lr=1.0,
                     validation_loss=0.0, test_loss=0.0)]
        self.assertEqual(select_learning_rates(rows, ["amsgrad"]), {"amsgrad": 0.001})

    def test_every_method_has_the_same_tuning_budget(self):
        self.assertEqual(len(METHODS), 18)
        self.assertEqual(len({m.name for m in METHODS}), 18)
        self.assertTrue(all(len(set(m.rates)) == 3 and min(m.rates) > 0 for m in METHODS))

    def exercise_methods(self, device):
        config = Config(vocab_size=8, n_layers=1, n_heads=2, d_model=16, d_ff=32, max_seq_len=8)
        for attention in ["softmax", "sparsemax"]:
            for method in METHODS:
                with self.subTest(device=device, attention=attention, optimizer=method.name):
                    model = make_model(config, attention, 0, device)
                    initial = model.embed.weight.detach().clone()
                    optimizer = make_optimizer(method.name, model, min(method.rates))
                    optimizer.zero_grad(set_to_none=True)
                    tokens = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]], device=device)
                    targets = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 0]], device=device)
                    loss = F.cross_entropy(model(tokens).flatten(0, 1), targets.flatten())
                    loss.backward()
                    self.assertTrue(all(p.grad is not None and bool(p.grad.isfinite().all())
                                        for p in model.parameters()))
                    optimizer.step()
                    self.assertTrue(all(bool(p.isfinite().all()) for p in model.parameters()))
                    self.assertFalse(torch.equal(initial, model.embed.weight))
                    optimizer.close()

    def test_every_optimizer_on_both_actual_models_cpu(self):
        self.exercise_methods("cpu")

    @unittest.skipUnless(os.environ.get("OPTIMIZER_BENCH_CUDA") == "1", "CUDA smoke is an explicit second phase")
    def test_every_optimizer_on_both_actual_models_cuda(self):
        self.assertTrue(torch.cuda.is_available(), "GPU benchmark requires actual CUDA access")
        self.exercise_methods("cuda")
