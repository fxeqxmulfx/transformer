"""Every added variant trains on mini GPT and reports honest held-out rollout."""

from dataclasses import replace
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from experiments.gpt_mini import GPTMini
from experiments.synthetic_trainers import training
from experiments.synthetic_trainers.config import ModelSpec, TrainConfig
from experiments.synthetic_trainers.data import build_split
from experiments.synthetic_trainers.metrics import evaluate
from experiments.synthetic_trainers.specs import CORE_TASKS, TASKS, TaskSpec
from experiments.synthetic_trainers.suite import expand_variants


def without_timing(value):
    if isinstance(value, dict):
        return {key: without_timing(item) for key, item in value.items() if key != "generation_seconds"}
    return value


class SuiteTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def config(self):
        return TrainConfig(steps=1, eval_every=1, batch_size=2, train_examples=2,
                           validation_examples=2, test_examples=2, eval_lengths=(16,), target=1.0)

    def test_every_added_variant_trains_evaluates_and_reloads_with_its_context_bound(self):
        model_spec = ModelSpec(width=8, layers=1, heads=1)
        with tempfile.TemporaryDirectory() as root:
            for task in TASKS:
                if task in CORE_TASKS:
                    continue
                spec = TaskSpec(task=task, length=8, symbols=32, number_limit=64)
                for variant in expand_variants(spec):
                    with self.subTest(variant=variant.run_name):
                        directory = Path(root) / variant.run_name
                        result = training.train_run(variant, model_spec, self.config(), directory)
                        self.assertEqual(result["steps_completed"], 1)
                        self.assertEqual(result["examples_seen"], 2)
                        self.assertEqual(result["variant"], variant.run_name)
                        self.assertTrue(math.isfinite(result["validation_final"]["loss"]))
                        self.assertIn("length-16", result["test"])
                        provenance = result["provenance"]
                        model = GPTMini(model_spec.reference_config(provenance["vocab_size"], provenance["context_length"]))
                        model.load_state_dict(torch.load(directory / "best.pt", weights_only=True))
                        rows = build_split(variant, "test", 0, 2).examples
                        observed = evaluate(model, rows, 2, spec=variant)
                        self.assertEqual(without_timing(observed), without_timing(result["test"]["in_distribution"]))
                        if variant.generative:
                            self.assertIn("teacher_forced", observed)
                            self.assertIn("generation_seconds", observed)
                            self.assertGreater(observed["generated_tokens"], 0)
                            self.assertGreater(observed["generation_padded_tokens_processed"], 0)
                        else:
                            self.assertNotIn("teacher_forced", observed)
                        saved = json.loads((directory / "result.json").read_text())
                        self.assertEqual(saved["variant"], variant.run_name)

    def test_addition_evaluates_hard_carry_at_training_and_ood_lengths(self):
        spec = TaskSpec(task="addition", length=4, number_limit=32, index_hints=True)
        config = replace(self.config(), eval_lengths=(8,))
        with tempfile.TemporaryDirectory() as root:
            result = training.train_run(spec, ModelSpec(width=8, layers=1, heads=1), config, root)
            self.assertEqual(set(result["test"]), {"in_distribution", "length-8", "hard-carry-length-4", "hard-carry-length-8"})
            specs = result["provenance"]["evaluation_specs"]
            self.assertEqual(specs["hard-carry-length-8"]["carry_length"], 8)
            self.assertTrue((Path(root) / "data" / "hard-carry-length-8" / "examples.jsonl").is_file())
            self.assertEqual(result["provenance"]["context_length"], 57)

    def test_and_evaluates_position_shift_separately_from_length_shift(self):
        spec = TaskSpec(task="boolean_and", length=8)
        with tempfile.TemporaryDirectory() as root:
            result = training.train_run(spec, ModelSpec(width=8, layers=1, heads=1), self.config(), root)
            self.assertEqual(set(result["test"]), {"in_distribution", "length-16", "position_shift", "position-shift-length-16"})
            specs = result["provenance"]["evaluation_specs"]
            self.assertEqual(specs["position_shift"]["length"], 8)
            self.assertEqual(specs["position_shift"]["and_region"], "late")
            self.assertNotEqual(result["split_fingerprints"]["in_distribution"], result["split_fingerprints"]["position_shift"])

    def test_crasp_report_persists_executable_formula_and_witnesses(self):
        spec = TaskSpec(task="crasp", length=8, formula_depth=3, formula_seed=5)
        with tempfile.TemporaryDirectory() as root:
            result = training.train_run(spec, ModelSpec(width=8, layers=1, heads=1), self.config(), root)
            program = result["provenance"]["program"]
            self.assertEqual(program["count_depth"], 3)
            self.assertIn("count", program["expression"])
            self.assertIn("op", program["ast"])
            self.assertEqual(len(program["witnesses"]), 2)
            saved = json.loads((Path(root) / "config.json").read_text())
            self.assertEqual(saved["program"]["ast"], json.loads(json.dumps(program["ast"])))

    def test_teacher_forced_success_does_not_trigger_a_free_generation_target(self):
        spec = TaskSpec(task="copy", length=4, symbols=4)
        metrics = {"sequence_accuracy": 0.0, "token_accuracy": 0.2, "balanced_accuracy": 0.2,
                   "loss": 0.01, "teacher_forced": {"sequence_accuracy": 1.0}}
        with tempfile.TemporaryDirectory() as root:
            with patch.object(training, "evaluate", return_value=metrics):
                result = training.train_run(spec, ModelSpec(width=8, layers=1, heads=1),
                                            replace(self.config(), eval_lengths=()), root)
            self.assertFalse(result["target_reached"])
            self.assertIsNone(result["time_to_target"])

    def test_final_answer_target_selects_the_best_final_answer_checkpoint(self):
        spec = TaskSpec(task="mode", length=4, symbols=4, scratchpad="counts", number_limit=16)
        def score(exact, final):
            return {"sequence_accuracy": exact, "final_answer_accuracy": final,
                    "balanced_accuracy": exact, "loss": 1.0}
        responses = (score(0.8, 0.8), score(0.1, 1.0), score(0.1, 1.0))
        config = replace(self.config(), eval_lengths=(), target_metric="final_answer_accuracy", target=0.9)
        with tempfile.TemporaryDirectory() as root:
            with patch.object(training, "evaluate", side_effect=responses):
                result = training.train_run(spec, ModelSpec(width=8, layers=1, heads=1), config, root)
            self.assertEqual(result["best_step"], 1)
            self.assertEqual(result["time_to_target"]["step"], 1)
            self.assertEqual(result["time_to_target"]["value"], 1.0)

    def test_invalid_ood_numeric_or_unique_capacity_fails_before_writing(self):
        for spec in (TaskSpec(task="copy", length=8, symbols=8, unique=True),
                     TaskSpec(task="histogram", length=8, number_limit=8)):
            with tempfile.TemporaryDirectory() as root:
                with self.assertRaises(ValueError):
                    training.train_run(spec, ModelSpec(width=8, layers=1, heads=1), self.config(), root)
                self.assertEqual(list(Path(root).iterdir()), [])

    def test_prefix_task_rejects_a_generated_final_answer_target(self):
        config = replace(self.config(), target_metric="final_answer_accuracy")
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(ValueError):
                training.train_run(TaskSpec(task="crasp", length=8), ModelSpec(width=8, layers=1, heads=1), config, root)
            self.assertEqual(list(Path(root).iterdir()), [])
