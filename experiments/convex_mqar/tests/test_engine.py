"""Loss accounting, optimizer coverage, warmup, and validation checkpoints."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

from convex_mqar import engine
from convex_mqar.convex import ConvexRecall
from convex_mqar.rope import RopeTransformer

from .support import batch, config


class EngineTests(unittest.TestCase):
    def test_evaluation_counts_queries_and_weights_the_partial_batch(self):
        class FixedModel(nn.Module):
            def forward(self, tokens, positions):
                table = torch.tensor([[4., 0., -1.], [3., 1., 0.], [0.2, 0., 1.]])
                return table[tokens.gather(1, positions)]

        tokens = torch.tensor([[0], [1], [2], [0], [1]])
        positions, labels = torch.zeros(5, 1, dtype=torch.long), torch.tensor([[0], [1], [2], [2], [0]])
        model = FixedModel().train()
        expected = model(tokens, positions).double().flatten(0, 1)
        loss = (expected.logsumexp(-1) - expected.gather(1, labels).squeeze(1)).mean()
        for size in (1, 2, 3, 10):
            result = engine.evaluate(model, (tokens, positions, labels), size, config())
            self.assertEqual(result["correct"], 3)
            self.assertEqual(result["query_count"], 5)
            self.assertEqual(result["accuracy"], 0.6)
            self.assertAlmostEqual(result["loss"], loss.item(), places=6)
            self.assertFalse(model.training)

    def test_convex_evaluation_counts_queries_and_reports_no_cross_entropy(self):
        data = batch(count=7)
        result = engine.evaluate(ConvexRecall(64, torch.ones(6)), data, 3, config(), convex=True)
        self.assertEqual(result["query_count"], 28)
        self.assertEqual(result["correct"], 28)
        self.assertEqual(result["accuracy"], 1)
        self.assertIsNone(result["loss"])

    def test_optimizer_covers_each_parameter_once_with_correct_decay(self):
        model = RopeTransformer(16, 8)
        optimizer = engine.optimizer_for(model, 0.003, 0.1)
        parameters = [p for group in optimizer.param_groups for p in group["params"]]
        self.assertEqual(len(parameters), len({id(p) for p in parameters}))
        self.assertEqual({id(p) for p in parameters}, {id(p) for p in model.parameters()})
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                self.assertEqual(group["weight_decay"], 0.1 if parameter.ndim >= 2 else 0.)
        self.assertEqual(sum(p is model.embedding.weight for p in parameters), 1)

    def test_optimizer_step_updates_the_tied_embedding_and_attention(self):
        torch.manual_seed(0)
        model = RopeTransformer(16, 8)
        optimizer = engine.optimizer_for(model, 0.003, 0.1)
        before = {name: p.detach().clone() for name, p in model.named_parameters()}
        tokens, positions, labels = batch(count=7, length=8, vocab=16)
        torch.nn.functional.cross_entropy(model(tokens, positions).flatten(0, 1), labels.flatten()).backward()
        optimizer.step()
        for name, parameter in model.named_parameters():
            with self.subTest(parameter=name):
                self.assertTrue(torch.isfinite(parameter).all())
                self.assertFalse(torch.equal(parameter, before[name]))

    def test_warmup_and_partial_batch_steps_match_the_training_budget(self):
        cfg = config(warmup_fraction=0.5)
        data = batch(count=67, length=8, vocab=16)
        recorded = []
        factory = engine.optimizer_for

        def instrument(model, lr, decay):
            optimizer = factory(model, lr, decay)
            step = optimizer.step

            def recording_step(*args, **kwargs):
                recorded.append([group["lr"] for group in optimizer.param_groups])
                return step(*args, **kwargs)

            optimizer.step = recording_step
            return optimizer

        with tempfile.TemporaryDirectory() as root, redirect_stdout(io.StringIO()):
            with patch.object(engine, "optimizer_for", side_effect=instrument):
                result, _ = engine.train_run(cfg, 8, 8, 0.003, data, data, root, Path(root) / "status.json")
            epochs = [json.loads(line) for line in (Path(root) / "epochs.jsonl").read_text().splitlines()]
        self.assertEqual(recorded, [[0.001, 0.001], [0.002, 0.002]] + [[0.003, 0.003]] * 4)
        self.assertEqual(result["epochs_completed"], 3)
        self.assertEqual(len(epochs), 3)
        self.assertEqual(epochs[-1]["epoch"], 3)

    def test_validation_accuracy_then_loss_selects_the_best_epoch(self):
        data = batch(count=5, length=8, vocab=16)
        validations = [{"accuracy": accuracy, "loss": loss} for accuracy, loss in
                       ((0.5, 2.), (0.5, 1.), (0.4, 0.1))]
        with tempfile.TemporaryDirectory() as root, redirect_stdout(io.StringIO()):
            with patch.object(engine, "evaluate", side_effect=validations):
                result, best = engine.train_run(config(), 8, 8, 0.003, data, data,
                                                root, Path(root) / "status.json")
            saved = torch.load(best, weights_only=True)
            current = torch.load(Path(root) / "current.pt", weights_only=False)
            self.assertEqual(result["best_epoch"], 2)
            self.assertEqual(result["validation"], validations[1])
            self.assertTrue(any(not torch.equal(saved[name], current["model"][name]) for name in saved))

    def test_nonfinite_training_loss_stops_without_publishing_results(self):
        class NonfiniteModel(nn.Module):
            def __init__(self, *args):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(()))

            def forward(self, tokens, positions):
                return self.weight * torch.full((*positions.shape, 16), torch.nan)

        data = batch(count=2, length=8, vocab=16)
        with tempfile.TemporaryDirectory() as root, redirect_stdout(io.StringIO()):
            with patch.object(engine, "RopeTransformer", NonfiniteModel):
                with self.assertRaisesRegex(RuntimeError, "Nonfinite loss"):
                    engine.train_run(config(), 8, 8, 0.003, data, data, root, Path(root) / "status.json")
            self.assertFalse((Path(root) / "result.json").exists())

    def test_training_reduces_loss_and_preserves_input_data(self):
        cfg = config(epochs=16)
        data = batch(count=32, length=8, vocab=16)
        originals = [tensor.clone() for tensor in data]
        torch.manual_seed(cfg.seed)
        initial = engine.evaluate(RopeTransformer(16, 8), data, 64, cfg)["loss"]
        with tempfile.TemporaryDirectory() as root, redirect_stdout(io.StringIO()):
            _, best = engine.train_run(cfg, 8, 8, 0.01, data, data, root, Path(root) / "status.json")
            model = RopeTransformer(16, 8)
            model.load_state_dict(torch.load(best, weights_only=True))
            final = engine.evaluate(model, data, 64, cfg)["loss"]
        self.assertLess(final, initial * 0.85)
        for tensor, original in zip(data, originals):
            torch.testing.assert_close(tensor, original)
