"""Independent simplex KKT oracle, derivative and GPTMini single-change checks."""

from copy import deepcopy
import itertools
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from experiments.gpt_mini import Config, GPTMini
from experiments.synthetic_trainers import sparsemax_attention
from experiments.synthetic_trainers.sparsemax_attention import causal_sparsemax, replace_attention


def exhaustive_simplex_projection(values):
    """Enumerate faces and solve their independent equality-constrained QP."""
    for size in range(1, len(values) + 1):
        for active in itertools.combinations(range(len(values)), size):
            threshold = (sum(values[i] for i in active) - 1) / size
            if (all(values[i] > threshold for i in active)
                    and all(values[i] <= threshold for i in range(len(values)) if i not in active)):
                return torch.tensor([values[i] - threshold if i in active else 0 for i in range(len(values))], dtype=torch.float64)
    raise AssertionError("No simplex face satisfies the KKT conditions")


class SparsemaxAttentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def model(self):
        torch.manual_seed(23)
        return GPTMini(Config(vocab_size=17, d_model=32, n_layers=2, n_heads=4, d_ff=128, max_seq_len=32)).double()

    def test_projection_matches_all_face_kkt_oracle_with_exact_causal_support(self):
        torch.manual_seed(41)
        scores = torch.randn(2, 6, 6, dtype=torch.float64)
        actual = causal_sparsemax(scores)
        for batch in range(2):
            for row in range(6):
                expected = exhaustive_simplex_projection(scores[batch, row, :row + 1].tolist())
                torch.testing.assert_close(actual[batch, row, :row + 1], expected, atol=1e-12, rtol=1e-12)
        self.assertEqual(actual.triu(1).count_nonzero().item(), 0)
        torch.testing.assert_close(actual.sum(-1), torch.ones(2, 6, dtype=torch.float64))
        self.assertTrue((actual >= 0).all())

    def test_projection_sparse_example_translation_and_low_precision_accumulation(self):
        scores = torch.tensor([[2., 99., 99.], [2., -3., 99.], [2., 1.6, -7.]], dtype=torch.float64)
        expected = torch.tensor([[1., 0., 0.], [1., 0., 0.], [.7, .3, 0.]], dtype=torch.float64)
        torch.testing.assert_close(causal_sparsemax(scores), expected)
        torch.testing.assert_close(causal_sparsemax(scores + 1e8), expected, atol=1e-8, rtol=1e-8)
        half = causal_sparsemax(scores.half())
        self.assertEqual(half.dtype, torch.float32)
        torch.testing.assert_close(half.sum(-1), torch.ones(3))

    def test_derivatives_match_finite_differences_and_future_gradients_vanish(self):
        torch.manual_seed(17)
        scores = torch.randn(2, 5, 5, dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(causal_sparsemax, (scores,), fast_mode=True))
        (causal_sparsemax(scores) * torch.randn_like(scores)).sum().backward()
        self.assertEqual(scores.grad.triu(1).count_nonzero().item(), 0)
        torch.testing.assert_close(scores.grad.sum(-1), torch.zeros(2, 5, dtype=torch.float64), atol=1e-12, rtol=0)

    def test_replacement_preserves_parameter_objects_rng_state_and_optimizer_bindings(self):
        model = self.model()
        optimizer = torch.optim.AdamW(model.parameters(), betas=(.9, .98), weight_decay=.1)
        before = {name: id(p) for name, p in model.named_parameters()}
        state, rng = deepcopy(model.state_dict()), torch.get_rng_state().clone()
        replace_attention(model)
        self.assertEqual({name: id(p) for name, p in model.named_parameters()}, before)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        self.assertEqual(set(model.state_dict()), set(state))
        for name, value in state.items():
            self.assertTrue(torch.equal(model.state_dict()[name], value), name)
        self.assertIs(model.embed.weight, model.unembed.weight)
        self.assertEqual({id(p) for group in optimizer.param_groups for p in group['params']}, set(before.values()))
        with self.assertRaises(ValueError):
            replace_attention(model)

    def test_restoring_softmax_matches_original_outputs_and_all_parameter_gradients(self):
        ordinary = self.model()
        sparse = replace_attention(deepcopy(ordinary))
        tokens = torch.randint(0, 17, (3, 7))
        labels = torch.randint(0, 17, (3, 7))
        expected = ordinary(tokens)
        F.cross_entropy(expected.flatten(0, 1), labels.flatten()).backward()

        def softmax(scores):
            future = torch.ones_like(scores, dtype=torch.bool).triu(1)
            return scores.masked_fill(future, -torch.inf).softmax(-1)

        with patch.object(sparsemax_attention, 'causal_sparsemax', side_effect=softmax):
            actual = sparse(tokens)
            F.cross_entropy(actual.flatten(0, 1), labels.flatten()).backward()
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
        for (name, p), (other, q) in zip(ordinary.named_parameters(), sparse.named_parameters()):
            self.assertEqual(name, other)
            torch.testing.assert_close(q.grad, p.grad, atol=1e-12, rtol=1e-12)

    def test_sparse_model_remains_causal_with_finite_full_backward(self):
        model = replace_attention(self.model())
        tokens = torch.randint(0, 17, (3, 9))
        expected = model(tokens)
        changed = tokens.clone()
        changed[:, 5:] = (changed[:, 5:] + 7) % 17
        torch.testing.assert_close(model(changed)[:, :5], expected[:, :5], atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(model(tokens[:, :5]), expected[:, :5], atol=1e-12, rtol=1e-12)
        F.cross_entropy(expected.flatten(0, 1), torch.randint(0, 17, (27,))).backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(parameter.grad.abs().sum().item(), 0, name)
