"""Padding, exact answers, class balance, and state-change accounting."""

import unittest

import torch
from torch import nn

from experiments.synthetic_trainers import vocabulary as v
from experiments.synthetic_trainers.metrics import Metrics, evaluate, masked_loss, validation_rank
from experiments.synthetic_trainers.records import Example, collate, example_from_targets


class MetricsTests(unittest.TestCase):
    def test_metrics_do_not_hide_errors_in_long_unchanged_label_runs(self):
        examples = [
            example_from_targets("blocks", (v.BOS, v.A, v.A, v.B), (v.IGNORE, v.REJECT, v.REJECT, v.ACCEPT)),
            example_from_targets("blocks", (v.BOS, v.A, v.B), (v.IGNORE, v.REJECT, v.ACCEPT)),
            example_from_targets("blocks", (v.BOS, v.NEUTRAL), (v.IGNORE, v.REJECT)),
        ]
        batch = collate(examples)
        predictions = torch.tensor([[0, v.REJECT, v.ACCEPT, v.ACCEPT],
                                    [0, v.REJECT, v.ACCEPT, 0], [0, v.ACCEPT, 0, 0]])
        logits = torch.zeros(3, 4, v.IDENTITY_BASE).scatter_(-1, predictions.unsqueeze(-1), 3.)
        metrics = Metrics()
        metrics.add(logits, batch)
        report = metrics.report()
        self.assertEqual(report["target_count"], 6)
        self.assertEqual(report["correct"], 4)
        self.assertEqual(report["token_accuracy"], 2 / 3)
        self.assertEqual(report["sequence_accuracy"], 1 / 3)
        self.assertEqual(report["balanced_accuracy"], 0.75)
        self.assertEqual(report["transition_count"], 5)
        self.assertEqual(report["transition_accuracy"], 0.8)

    def test_loss_has_no_gradient_at_unsupervised_or_padded_positions(self):
        logits = torch.randn(2, 4, 3, requires_grad=True)
        targets = torch.tensor([[v.IGNORE, 1, 2, v.IGNORE], [v.IGNORE, 0, v.IGNORE, v.IGNORE]])
        masked_loss(logits, targets).backward()
        self.assertTrue(torch.equal(logits.grad[targets == v.IGNORE], torch.zeros_like(logits.grad[targets == v.IGNORE])))
        self.assertGreater(float(logits.grad[targets != v.IGNORE].abs().sum()), 0)

    def test_partial_batches_have_identical_weighted_loss_and_accuracy(self):
        class Fixed(nn.Module):
            def forward(self, tokens):
                table = torch.tensor([[3., 1., 0.], [0., 2., 1.], [0.2, 0., 1.]])
                return table[tokens]

        examples = [Example("lookup", (token,), (target,), (True,))
                    for token, target in ((0, 0), (1, 0), (2, 2), (0, 1), (1, 1))]
        logits = Fixed()(collate(examples).tokens).double().squeeze(1)
        targets = torch.tensor([0, 0, 2, 1, 1])
        expected = float((logits.logsumexp(-1) - logits.gather(1, targets[:, None]).squeeze(1)).mean())
        for size in (1, 2, 3, 8):
            model = Fixed().train()
            result = evaluate(model, examples, size)
            self.assertTrue(model.training)
            self.assertEqual(result["correct"], 3)
            self.assertEqual(result["target_count"], 5)
            self.assertEqual(result["sequence_accuracy"], 0.6)
            self.assertAlmostEqual(result["loss"], expected, places=6)

    def test_right_padding_does_not_create_targets_or_change_masks(self):
        first = Example("lookup", (v.BOS, v.QUERY, 17), (v.IGNORE, v.IGNORE, 18), (False, False, True))
        second = Example("lookup", (v.BOS, v.FILL, v.FILL, v.QUERY, 18),
                         (v.IGNORE, v.IGNORE, v.IGNORE, v.IGNORE, 17), (False, False, False, False, True))
        batch = collate((first, second))
        self.assertEqual(batch.tokens[0, 3:].tolist(), [v.PAD, v.PAD])
        self.assertEqual(batch.targets[0, 3:].tolist(), [v.IGNORE, v.IGNORE])
        self.assertFalse(batch.changes[0, 3:].any())

    def test_empty_supervision_and_nonfinite_metrics_fail(self):
        with self.assertRaises(ValueError):
            masked_loss(torch.zeros(2, 3, 4), torch.full((2, 3), v.IGNORE))
        with self.assertRaises(ValueError):
            collate([])
        with self.assertRaises(ValueError):
            Metrics().report()
        batch = collate([Example("lookup", (17,), (18,), (True,))])
        with self.assertRaisesRegex(RuntimeError, "Nonfinite"):
            Metrics().add(torch.full((1, 1, 20), float("nan")), batch)

    def test_validation_ranking_uses_exactness_then_balance_then_loss(self):
        def metric(exact, balanced, loss):
            return {"sequence_accuracy": exact, "balanced_accuracy": balanced, "loss": loss}
        self.assertGreater(validation_rank(metric(0.6, 0.2, 8)), validation_rank(metric(0.5, 0.9, 1)))
        self.assertGreater(validation_rank(metric(0.5, 0.8, 8)), validation_rank(metric(0.5, 0.7, 1)))
        self.assertGreater(validation_rank(metric(0.5, 0.8, 1)), validation_rank(metric(0.5, 0.8, 2)))
