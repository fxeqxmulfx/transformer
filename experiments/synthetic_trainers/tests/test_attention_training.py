"""Native optimizer trajectory, tagged checkpoint and exact sparsemax resume."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from experiments.synthetic_trainers import attention_training as attention
from experiments.synthetic_trainers.paper_reproduction import grokking
from experiments.synthetic_trainers.paper_reproduction.diagnostics import DiagnosticsConfig


class AttentionTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def config(self, normalization="softmax", **kwargs):
        settings = dict(model="gptmini", optimizer="adamw", prime=7, train_fraction=.5,
                        steps=20, batch_size=4, eval_every=5, learning_rate=.0003,
                        weight_decay=.1, device="cpu", attention_normalization=normalization)
        return attention.AttentionRunConfig(**(settings | kwargs))

    def equal_states(self, left, right):
        if isinstance(left, torch.Tensor):
            self.assertTrue(torch.equal(left, right))
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.equal_states(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.equal_states(a, b)
        else:
            self.assertEqual(left, right)

    def check_checkpoints(self, left, right):
        for key in ("model", "optimizer", "step", "examples_seen", "last_batch_size",
                    "batch_generator_state", "permutation", "cursor"):
            self.equal_states(left[key], right[key])

    def test_softmax_keeps_the_original_full_width_native_adamw_trajectory(self):
        config = self.config()
        plain = grokking.RunConfig(**{k: v for k, v in config.__dict__.items() if k != "attention_normalization"})
        diagnostics = DiagnosticsConfig(5, True, True)
        with tempfile.TemporaryDirectory() as temporary:
            base, wrapped = Path(temporary)/"ordinary", Path(temporary)/"wrapped"
            expected = grokking.train(plain, base, diagnostics=diagnostics)
            actual = attention.train(config, wrapped, diagnostics=diagnostics)
            self.check_checkpoints(torch.load(base/"checkpoint.pt", weights_only=True),
                                   torch.load(wrapped/"checkpoint.pt", weights_only=True))
            for before, after in zip(expected["history"], actual["history"]):
                for key in ("step", "train", "heldout", "epochs_seen", "last_batch_size"):
                    self.assertEqual(before[key], after[key])
            for name in ("gradients.jsonl", "diagnostics.jsonl", "probes.jsonl"):
                if name == "probes.jsonl":
                    def rows(path):
                        return [{k:v for k,v in json.loads(line).items() if k not in ("training_seconds","wall_seconds")}
                                for line in path.read_text().splitlines()]
                    self.assertEqual(rows(base/name), rows(wrapped/name))
                else:
                    self.assertEqual((base/name).read_bytes(), (wrapped/name).read_bytes())
            self.assertEqual(actual["plan"]["config"]["attention_normalization"], "softmax")
            self.assertEqual(actual["plan"]["source_hashes"], attention.training_sources())

    def test_sparsemax_resume_preserves_native_buffers_sampling_and_warmup(self):
        config = self.config("sparsemax")
        diagnostics = DiagnosticsConfig(5, True, True)
        with tempfile.TemporaryDirectory() as temporary:
            fresh, extended = Path(temporary)/"fresh", Path(temporary)/"extended"
            uninterrupted = attention.train(config, fresh, diagnostics=diagnostics)
            attention.train(replace(config, steps=10), extended, diagnostics=diagnostics)
            resumed = attention.train(config, extended, resume=True, diagnostics=diagnostics)
            self.check_checkpoints(torch.load(fresh/"checkpoint.pt", weights_only=True),
                                   torch.load(extended/"checkpoint.pt", weights_only=True))
            for before, after in zip(uninterrupted["history"], resumed["history"]):
                self.assertEqual(before["train"], after["train"])
                self.assertEqual(before["heldout"], after["heldout"])
            for name in ("gradients.jsonl", "diagnostics.jsonl"):
                self.assertEqual((fresh/name).read_bytes(), (extended/name).read_bytes())
            native = torch.load(extended/"checkpoint.pt", weights_only=True)["optimizer"]
            self.assertTrue(all(float(state["step"]) == 20 for state in native["state"].values()))

    def test_cross_normalizer_and_ordinary_resume_are_rejected_before_load(self):
        config = self.config("sparsemax", steps=2, eval_every=1)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)/"run"
            attention.train(config, directory)
            before = {p.name:p.read_bytes() for p in directory.iterdir() if p.is_file()}
            with patch.object(grokking.torch, "load", side_effect=AssertionError("Must reject before loading")):
                with self.assertRaisesRegex(ValueError, "original update budget"):
                    attention.train(replace(config, attention_normalization="softmax"), directory, resume=True)
                plain = grokking.RunConfig(**{k:v for k,v in config.__dict__.items() if k != "attention_normalization"})
                with self.assertRaisesRegex(ValueError, "sources"):
                    grokking.train(plain, directory, resume=True)
            self.assertEqual(before, {p.name:p.read_bytes() for p in directory.iterdir() if p.is_file()})

    def test_factory_and_sources_are_restored_after_training_exception(self):
        config = self.config()
        factory, hashes = grokking.make_model, grokking.source_hashes
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(grokking, "evaluate", side_effect=RuntimeError("Interrupted CPU fixture")):
                with self.assertRaisesRegex(RuntimeError, "Interrupted CPU fixture"):
                    attention.train(config, Path(temporary)/"run")
        self.assertIs(grokking.make_model, factory)
        self.assertIs(grokking.source_hashes, hashes)

    def test_invalid_normalizers_optimizers_and_total_update_caps_are_rejected(self):
        for setting in ({"attention_normalization":"entmax"},{"optimizer":"amsgradw"},
                        {"model":"reference"},{"steps":300001}):
            with self.assertRaises(ValueError):
                self.config(**setting)
        with self.assertRaises(TypeError):
            attention.make_model(grokking.RunConfig(model="gptmini",device="cpu"),17)
        with self.assertRaises(TypeError):
            attention.train(grokking.RunConfig(model="gptmini",device="cpu"),"unused")
