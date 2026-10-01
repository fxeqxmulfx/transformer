"""Compiled stopping/checkpoint integration contracts before long GPU runs."""

import json
import os
from pathlib import Path
import tempfile
import unittest

import torch

from experiments.gpt_mini import Config
from experiments.optimizer_benchmark.attention import make_model
from experiments.optimizer_benchmark.data import TextData
from experiments.optimizer_benchmark.runner import state_hash
from experiments.patience_benchmark import training as eager_training
from experiments.patience_benchmark.stopping import StopConfig, choose_best_step
from experiments.compiled_benchmark.training import train_one
from experiments.compiled_benchmark.validation import check_checkpoint


@unittest.skipUnless(os.environ.get("OPTIMIZER_BENCH_CUDA") == "1", "CUDA compile is explicit")
class CompiledTrainingTests(unittest.TestCase):
    def test_best_restore_and_fresh_compiled_checkpoint_reload(self):
        torch.set_num_threads(4)
        cfg = Config(vocab_size=8, n_layers=1, n_heads=2, d_model=16, d_ff=32, max_seq_len=8)
        stopping = StopConfig(every=2, patience=3, min_steps=1, max_steps=6)
        original_factory, original_evaluate = eager_training.make_model, eager_training.evaluate
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            text = directory / "text.txt"
            text.write_text("abcdefgh" * 200)
            data = TextData.load(text, "cuda")
            for attention in ("softmax", "sparsemax"):
                for method, rate in (("magma_muon", 0.01), ("adafisher", 0.0001)):
                    with self.subTest(attention=attention, method=method):
                        row = train_one(cfg, data, attention, method, rate, 0, 2, stopping, directory)
                        self.assertEqual(row["status"], "ok", row.get("recovery_error"))
                        self.assertEqual(row["actual_steps"], 6)
                        self.assertEqual(row["test_evaluations"], 1)
                        self.assertEqual(row["best_step"], choose_best_step(row["curves"]))
                        reference = make_model(cfg, attention, 0, "cpu")
                        self.assertEqual(row["initial_sha256"], state_hash(reference))
                        self.assertGreater(row["compile"]["final"]["recorded_graph_nodes"], 0)
                        self.assertEqual(row["compile"]["fullgraph"], method != "adafisher")
                        self.assertEqual(set(row["compile"]["cold_forward_seconds"]), {"train", "inference"})
                        check = check_checkpoint(row, cfg, data, 2)
                        self.assertEqual(check["test_loss"], row["test_loss"])
                        json.dumps(row, allow_nan=False)
                        self.assertIs(eager_training.make_model, original_factory)
                        self.assertIs(eager_training.evaluate, original_evaluate)
