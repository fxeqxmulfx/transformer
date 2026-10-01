"""All modes train, select only on validation, and publish reloadable results."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

from experiments.gpt_mini import GPTMini
from experiments.synthetic_trainers import training
from experiments.synthetic_trainers.config import ModelSpec, TrainConfig
from experiments.synthetic_trainers.data import build_split
from experiments.synthetic_trainers.metrics import evaluate
from experiments.synthetic_trainers.specs import CORE_TASKS, TaskSpec


class TinyCausalModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.output = nn.Linear(config.d_model, config.vocab_size)

    def forward(self, tokens):
        return self.output(self.embedding(tokens).cumsum(dim=1))


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def config(self):
        return replace(TrainConfig(), steps=3, eval_every=2, batch_size=3,
                       train_examples=5, validation_examples=5, test_examples=5,
                       eval_lengths=(48,), target=1.0)

    def test_all_modes_complete_training_and_reload_selected_checkpoint(self):
        model_spec = ModelSpec(width=16, layers=1, heads=2)
        config = self.config()
        for task in CORE_TASKS:
            spec = TaskSpec(task=task, length=24, min_length=22, pairs=4, queries=2)
            with self.subTest(task=task), tempfile.TemporaryDirectory() as root:
                result = training.train_run(spec, model_spec, config, root)
                self.assertEqual(result["steps_completed"], 3)
                self.assertEqual(result["examples_seen"], 8)  # partial batch: 3 + 2 + 3
                self.assertEqual(set(result["test"]), {"in_distribution", "length-48"})
                history = [json.loads(line) for line in (Path(root) / "history.jsonl").read_text().splitlines()]
                self.assertEqual([row["step"] for row in history], [0, 2, 3])
                self.assertGreaterEqual(result["training_wall_seconds"], result["training_seconds"])
                model = GPTMini(model_spec.reference_config(spec.vocab_size, 48))
                model.load_state_dict(torch.load(Path(root) / "best.pt", weights_only=True))
                test = build_split(spec, "test", config.data_seed, config.test_examples)
                observed = evaluate(model, test.examples, config.batch_size)
                self.assertEqual(result["test"]["in_distribution"], observed)
                persisted = json.loads((Path(root) / "result.json").read_text())
                self.assertEqual(persisted["split_fingerprints"]["in_distribution"], test.fingerprint)
                self.assertTrue((Path(root) / "final.pt").is_file())
                self.assertTrue(result["provenance"]["source_hashes"])

    def test_data_seeds_are_shared_across_different_model_seeds(self):
        spec = TaskSpec(length=24, pairs=4, queries=2)
        model_spec = ModelSpec(width=8, layers=1, heads=1)
        with tempfile.TemporaryDirectory() as root:
            first = training.train_run(spec, model_spec, self.config(), Path(root) / "first")
            second = training.train_run(spec, model_spec, replace(self.config(), seed=8), Path(root) / "second")
            self.assertEqual(first["split_fingerprints"], second["split_fingerprints"])
            a = torch.load(Path(root) / "first" / "final.pt", weights_only=True)
            b = torch.load(Path(root) / "second" / "final.pt", weights_only=True)
            self.assertTrue(any(not torch.equal(a[key], b[key]) for key in a))

    def test_test_data_is_generated_only_after_validation_selection(self):
        spec = TaskSpec(length=24, pairs=4, queries=2)
        events = []
        original = training.build_split

        def record_build(spec, name, seed, count):
            events.append(name)
            return original(spec, name, seed, count)

        with tempfile.TemporaryDirectory() as root:
            with patch.object(training, "build_split", side_effect=record_build):
                training.train_run(spec, ModelSpec(width=8, layers=1, heads=1), self.config(), root,
                                   progress=lambda row: events.append(f"validation-step-{row['step']}"))
        self.assertGreater(events.index("test"), events.index("validation-step-3"))
        self.assertEqual(events.count("test"), 2)

    def test_checkpoint_and_time_to_target_use_validation_instead_of_test(self):
        def metrics(exact, loss):
            return {"sequence_accuracy": exact, "balanced_accuracy": exact,
                    "token_accuracy": exact, "loss": loss}

        config = replace(self.config(), steps=2, eval_every=1, target=0.75, eval_lengths=())
        spec = TaskSpec(length=24, pairs=4, queries=2)
        responses = [metrics(0.1, 5), metrics(0.8, 3), metrics(0.2, 1), metrics(1.0, 0)]
        with tempfile.TemporaryDirectory() as root:
            with patch.object(training, "evaluate", side_effect=responses):
                result = training.train_run(spec, ModelSpec(width=8, layers=1, heads=1), config, root)
            self.assertEqual(result["best_step"], 1)
            self.assertEqual(result["time_to_target"]["step"], 1)
            self.assertEqual(result["time_to_target"]["examples_seen"], 3)
            self.assertEqual(result["steps_completed"], 2)  # target does not shorten a fixed budget
            best = torch.load(Path(root) / "best.pt", weights_only=True)
            final = torch.load(Path(root) / "final.pt", weights_only=True)
            self.assertTrue(any(not torch.equal(best[key], final[key]) for key in best))

    def test_unreached_target_is_not_reported_as_success_from_test_score(self):
        metrics = {"sequence_accuracy": 0., "balanced_accuracy": 0., "token_accuracy": 0., "loss": 1.}
        spec = TaskSpec(length=24, pairs=4, queries=2)
        with tempfile.TemporaryDirectory() as root:
            with patch.object(training, "evaluate", return_value=metrics):
                result = training.train_run(spec, ModelSpec(width=8, layers=1, heads=1), self.config(), root)
            self.assertFalse(result["target_reached"])
            self.assertIsNone(result["time_to_target"])

    def test_optional_early_stop_keeps_the_observed_target_budget(self):
        metrics = {"sequence_accuracy": 1., "balanced_accuracy": 1., "token_accuracy": 1., "loss": 1.}
        spec = TaskSpec(length=24, pairs=4, queries=2)
        with tempfile.TemporaryDirectory() as root:
            with patch.object(training, "evaluate", return_value=metrics):
                result = training.train_run(spec, ModelSpec(width=8, layers=1, heads=1),
                                            replace(self.config(), stop_at_target=True), root)
            self.assertEqual(result["steps_completed"], 0)
            self.assertEqual(result["time_to_target"]["step"], 0)
            self.assertEqual(result["examples_seen"], 0)

    def test_custom_causal_architecture_uses_the_same_task_and_protocol(self):
        spec = TaskSpec(length=24, pairs=4, queries=2)
        with tempfile.TemporaryDirectory() as root:
            result = training.train_run(spec, ModelSpec(width=8, layers=1, heads=1), self.config(), root,
                                        model_factory=TinyCausalModel)
            self.assertIn("TinyCausalModel", result["provenance"]["factory"])
            self.assertEqual(result["steps_completed"], 3)

    def test_nonfinite_model_does_not_publish_a_completed_report(self):
        class Nonfinite(TinyCausalModel):
            def forward(self, tokens):
                return super().forward(tokens) * float("nan")

        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(RuntimeError, "Nonfinite"):
                training.train_run(TaskSpec(length=24, pairs=4, queries=2),
                                   ModelSpec(width=8, layers=1, heads=1), self.config(), root,
                                   model_factory=Nonfinite)
            self.assertFalse((Path(root) / "result.json").exists())

    def test_existing_artifacts_and_non_ood_lengths_are_rejected(self):
        spec = TaskSpec(length=24, pairs=4, queries=2)
        model_spec = ModelSpec(width=8, layers=1, heads=1)
        with tempfile.TemporaryDirectory() as root:
            sentinel = Path(root) / "do-not-overwrite.txt"
            sentinel.write_text("existing work")
            with self.assertRaises(FileExistsError):
                training.train_run(spec, model_spec, self.config(), root)
            self.assertEqual(sentinel.read_text(), "existing work")
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(ValueError):
                training.train_run(spec, model_spec, replace(self.config(), eval_lengths=(24,)), root)
