"""Actual complementary updates, normalizer pairing and portable tag integrity."""

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import torch

from experiments.gpt_mini import GPTMini
from experiments.synthetic_trainers import complementary_attention as adapter
from experiments.synthetic_trainers import complementary_training
from experiments.synthetic_trainers.complementary_attention_report import paired_quality, verify_attention_run
from experiments.synthetic_trainers.complementary_config import ComplementaryConfig
from experiments.synthetic_trainers.complementary_report import hashes
from experiments.synthetic_trainers.config import ModelSpec
from experiments.synthetic_trainers.runtime import write_json
from experiments.synthetic_trainers.specs import TaskSpec


def without_time(value):
    if isinstance(value, dict):
        return {key: without_time(item) for key, item in value.items()
                if key not in ("training_seconds", "wall_seconds", "generation_seconds")}
    if isinstance(value, list):
        return [without_time(item) for item in value]
    return value


def equal_tensors(test, left, right):
    if isinstance(left, torch.Tensor):
        test.assertTrue(torch.equal(left, right))
    elif isinstance(left, dict):
        test.assertEqual(left.keys(), right.keys())
        for key in left:
            equal_tensors(test, left[key], right[key])
    elif isinstance(left, (tuple, list)):
        test.assertEqual(len(left), len(right))
        for first, second in zip(left, right):
            equal_tensors(test, first, second)
    else:
        test.assertEqual(left, right)


class ComplementaryAttentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def config(self):
        return ComplementaryConfig(steps=16, eval_every=4, batch_size=2,
            train_examples=3, validation_examples=3, test_examples=3, eval_lengths=(8,), target=1.0)

    def test_both_factories_preserve_real_scientific_shape_initialization_and_rng(self):
        model = ModelSpec(width=128, layers=2, heads=4, init_std=.02)
        cfg = model.reference_config(32, 48)
        torch.manual_seed(11)
        original = GPTMini(cfg); rng = torch.get_rng_state().clone()
        for factory in (adapter.softmax_model, adapter.sparsemax_model):
            torch.manual_seed(11)
            actual = factory(cfg)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            equal_tensors(self, original.state_dict(), actual.state_dict())
            self.assertEqual(list(dict(original.named_parameters())), list(dict(actual.named_parameters())))
            self.assertIs(actual.embed.weight, actual.unembed.weight)

    def test_softmax_retains_exact_native_updates_moments_sampling_and_scores(self):
        spec = TaskSpec(task="copy", length=4, symbols=8)
        model = ModelSpec(width=8, layers=1, heads=1, init_std=.02)
        for schedule in ("constant", "cosine_tail"):
            cfg = replace(self.config(), learning_rate_schedule=schedule, anneal_start=10, anneal_end=14)
            with self.subTest(schedule=schedule), tempfile.TemporaryDirectory() as root:
                parent, actual = Path(root) / "parent", Path(root) / "actual"
                first = complementary_training.train(cfg, spec, model, parent)
                second = adapter.train(cfg, spec, model, actual, attention_normalization="softmax")
                for name in ("final.pt", "best.pt", "optimizer-final.pt"):
                    equal_tensors(self, torch.load(parent / name, weights_only=True), torch.load(actual / name, weights_only=True))
                self.assertEqual((parent / "update-rates.jsonl").read_bytes(), (actual / "update-rates.jsonl").read_bytes())
                histories = [[without_time(json.loads(line)) for line in (path / "history.jsonl").read_text().splitlines()]
                             for path in (parent, actual)]
                self.assertEqual(*histories)
                self.assertEqual(without_time(first["test"]), without_time(second["test"]))
                self.assertEqual(without_time(first["test_final"]), without_time(second["test_final"]))

    def test_complete_real_pair_and_rehashed_tag_or_config_forgeries(self):
        cfg = self.config(); spec = TaskSpec(task="copy", length=4, symbols=8)
        model = ModelSpec(width=8, layers=1, heads=1, init_std=.02)
        with tempfile.TemporaryDirectory() as root:
            paths = [Path(root) / name for name in ("softmax", "sparsemax")]
            for normalizer, path in zip(("softmax", "sparsemax"), paths):
                adapter.train(cfg, spec, model, path, attention_normalization=normalizer)
                write_json(path / "artifact-hashes.json", {"files": hashes(path)})
                self.assertEqual(verify_attention_run(path, attention_normalization=normalizer)["steps_completed"], 16)
            self.assertFalse(paired_quality(*paths)["scientific_effect_certified"])
            code = ("from experiments.synthetic_trainers.complementary_attention_report import paired_quality; "
                    "import sys,json; print(json.dumps(paired_quality(*sys.argv[1:]))); assert 'torch' not in sys.modules")
            subprocess.run(["python3", "-c", code, *map(str, paths)], check=True, capture_output=True, text=True)
            path = paths[1] / "result.json"; original = path.read_bytes()
            result = json.loads(original); result["complementary_attention"]["attention_normalization"] = "softmax"
            write_json(path, result); write_json(paths[1] / "artifact-hashes.json", {"files": hashes(paths[1])})
            with self.assertRaisesRegex(ValueError, "declared normalizer"):
                paired_quality(*paths)
            path.write_bytes(original); write_json(paths[1] / "artifact-hashes.json", {"files": hashes(paths[1])})
            with self.assertRaisesRegex(ValueError, "declared normalizer"):
                verify_attention_run(paths[1], attention_normalization="softmax")
            result = json.loads(original)
            result["provenance"]["training"]["seed"] = 1
            result["complementary_protocol"]["config"]["seed"] = 1
            write_json(path, result); write_json(paths[1] / "config.json", result["provenance"])
            write_json(paths[1] / "artifact-hashes.json", {"files": hashes(paths[1])})
            with self.assertRaisesRegex(ValueError, "settings"):
                paired_quality(*paths)

    def test_invalid_or_interrupted_run_never_overwrites_and_restores_optimizer(self):
        from experiments.synthetic_trainers import training
        cfg = self.config(); spec = TaskSpec(task="copy", length=4, symbols=8)
        model = ModelSpec(width=8, layers=1, heads=1, init_std=.02)
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "run"; original = training.optimizer_for
            with patch.object(adapter, "softmax_model", side_effect=AssertionError("no model")):
                with self.assertRaises(ValueError):
                    adapter.train(cfg, spec, model, path, attention_normalization="unknown")
            self.assertFalse(path.exists())
            def interrupt(row):
                if row["step"] == 4:
                    raise RuntimeError("intentional interruption")
            with self.assertRaisesRegex(RuntimeError, "intentional interruption"):
                adapter.train(cfg, spec, model, path, attention_normalization="sparsemax", progress=interrupt)
            saved = {str(p.relative_to(path)): p.read_bytes() for p in path.rglob("*") if p.is_file()}
            with self.assertRaises(FileExistsError):
                adapter.train(cfg, spec, model, path, attention_normalization="sparsemax")
            self.assertEqual(saved, {str(p.relative_to(path)): p.read_bytes() for p in path.rglob("*") if p.is_file()})
            self.assertIs(training.optimizer_for, original)
            self.assertEqual(len((path / "update-rates.jsonl").read_text().splitlines()), 4)


if __name__ == "__main__":
    unittest.main()
