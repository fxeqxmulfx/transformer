"""Recovery and best-validation selection contracts before training code."""

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from experiments.gpt_mini import Config
from experiments.optimizer_benchmark.data import TextData, training_starts
from experiments.optimizer_benchmark.runner import state_hash
from experiments.optimizer_benchmark.attention import make_model
from experiments.patience_benchmark.registry import METHODS, selected_rates
from experiments.patience_benchmark.stopping import StopConfig
from experiments.patience_benchmark.training import train_one


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def data(self, directory, device):
        path = Path(directory) / "text.txt"
        path.write_text("abcdefgh" * 200)
        return TextData.load(path, device)

    def test_all_24_methods_have_validation_selected_rates(self):
        names = {m.name for m in METHODS}
        self.assertEqual(len(names), 24)
        for rates in selected_rates().values():
            self.assertEqual(set(rates), names)
            self.assertTrue(all(rate is not None and rate > 0 for rate in rates.values()))

    def test_long_batch_stream_preserves_previous_prefix(self):
        for seed in range(3):
            short = training_starts(10000, 64, 32, 1000, seed)
            long = training_starts(10000, 64, 32, 20000, seed)
            torch.testing.assert_close(short, long[:1000], atol=0, rtol=0)

    def test_restores_exact_validation_best_and_evaluates_test_once(self):
        cfg = Config(vocab_size=8, n_layers=1, n_heads=2, d_model=16, d_ff=32, max_seq_len=8)
        stopping = StopConfig(every=1, patience=2, min_delta=0.001, min_steps=1,
                              max_steps=8, divergence_delta=0.1, divergence_patience=2)
        seen = []
        losses = iter((2.0, 1.0, 1.2, 1.3, 1.0, 0.9))

        def evaluation(model, tokens, context, batch):
            seen.append(state_hash(model))
            return next(losses)

        with tempfile.TemporaryDirectory() as directory:
            data = self.data(directory, "cpu")
            with patch("experiments.patience_benchmark.training.evaluate", side_effect=evaluation):
                row = train_one(cfg, data, "softmax", "magma_adamw", 0.001, 0, 2,
                                stopping, Path(directory), device="cpu")
            self.assertEqual(row["stop_reason"], "validation_divergence")
            self.assertEqual((row["actual_steps"], row["best_step"]), (3, 1))
            self.assertEqual((row["validation_loss"], row["test_loss"]), (1.0, 0.9))
            self.assertEqual(row["test_evaluations"], 1)
            self.assertEqual(len(seen), 6)
            self.assertEqual(seen[1], seen[-2])
            self.assertEqual(seen[1], seen[-1])
            self.assertNotEqual(seen[1], seen[3])
            self.assertEqual(row["magma"]["mask_draws"], 12)
            checkpoint = torch.load(row["checkpoint"], weights_only=True)
            model = make_model(cfg, "softmax", 0)
            model.load_state_dict(checkpoint["model"])
            self.assertEqual(state_hash(model), seen[1])

    def test_nonfinite_validation_preserves_usable_checkpoint_and_json(self):
        cfg = Config(vocab_size=8, n_layers=1, n_heads=2, d_model=16, d_ff=32, max_seq_len=8)
        stopping = StopConfig(every=1, patience=2, min_steps=4, max_steps=8)
        with tempfile.TemporaryDirectory() as directory:
            data = self.data(directory, "cpu")
            with patch("experiments.patience_benchmark.training.evaluate", side_effect=(2.0, float("nan"), 2.0, 2.1)):
                row = train_one(cfg, data, "softmax", "sgd", 0.1, 0, 2, stopping,
                                Path(directory), device="cpu")
            self.assertEqual(row["status"], "recovered")
            self.assertEqual(row["stop_reason"], "nonfinite_validation")
            self.assertEqual((row["actual_steps"], row["best_step"]), (1, 0))
            self.assertIsNone(row["curves"][-1]["validation_loss"])
            import json
            json.dumps(row, allow_nan=False)

    @unittest.skipUnless(os.environ.get("OPTIMIZER_BENCH_CUDA") == "1", "CUDA is an explicit phase")
    def test_gpu_training_loop_on_both_attentions_and_both_factories(self):
        self.assertTrue(torch.cuda.is_available())
        cfg = Config(vocab_size=8, n_layers=1, n_heads=2, d_model=16, d_ff=32, max_seq_len=8)
        stopping = StopConfig(every=1, patience=3, min_steps=1, max_steps=2)
        with tempfile.TemporaryDirectory() as directory:
            data = self.data(directory, "cuda")
            for attention in ("softmax", "sparsemax"):
                for name, lr in (("adamw", 0.001), ("magma_muon", 0.01), ("dash_ndb", 0.001), ("adafisher", 0.0001)):
                    with self.subTest(attention=attention, method=name):
                        row = train_one(cfg, data, attention, name, lr, 0, 2, stopping,
                                        Path(directory), device="cuda")
                        self.assertEqual(row["status"], "ok")
                        self.assertEqual(row["stop_reason"], "max_steps")
                        self.assertEqual(row["test_evaluations"], 1)
                        self.assertEqual(row["actual_steps"], 2)
