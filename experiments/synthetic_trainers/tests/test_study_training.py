"""Both checkpoints, paired grids, finite budgets, and the public study commands."""

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from experiments.gpt_mini import GPTMini
from experiments.synthetic_trainers.cli import main
from experiments.synthetic_trainers.config import ModelSpec, TrainConfig
from experiments.synthetic_trainers.corpus import study_pool
from experiments.synthetic_trainers.metrics import evaluate
from experiments.synthetic_trainers.specs import TaskSpec
from experiments.synthetic_trainers.sweeps import SweepConfig, run_name, run_sweep, summarize
from experiments.synthetic_trainers.training import train_run
from experiments.synthetic_trainers.tests.test_suite_training import without_timing


class StudyTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def config(self):
        return TrainConfig(study="double_descent", steps=2, eval_every=1, batch_size=2,
                           train_examples=3, validation_examples=3, test_examples=3, eval_lengths=(8,))

    def model(self):
        return ModelSpec(width=8, layers=1, heads=1)

    def test_noise_and_ood_curves_are_saved_and_final_checkpoint_reloads(self):
        spec = TaskSpec(task="copy", length=4, symbols=8)
        config = replace(self.config(), label_noise=1.0, split_policy="disjoint")
        with tempfile.TemporaryDirectory() as root:
            result = train_run(spec, self.model(), config, root)
            directory = Path(root)
            rows = [json.loads(line) for line in (directory / "history.jsonl").read_text().splitlines()]
            self.assertEqual([row["step"] for row in rows], [0, 1, 2])
            self.assertEqual([row["epochs_seen"] for row in rows], [0, 2 / 3, 1])
            self.assertTrue(all("noise_fit" in row and "length-8" in row["validation_ood"] for row in rows))
            self.assertEqual(result["study"]["noise"]["realized_rate"], 1)
            self.assertEqual(result["study"]["final_checkpoint"]["clean_answer_entropy_given_input_bits"], 0)
            self.assertNotEqual(result["split_fingerprints"]["train"], result["split_fingerprints"]["train_clean"])
            self.assertEqual(result["provenance"]["parameter_storage_bits"], 32 * result["parameters"])
            model = GPTMini(self.model().reference_config(result["provenance"]["vocab_size"], result["provenance"]["context_length"]))
            model.load_state_dict(torch.load(directory / "final.pt", weights_only=True))
            pool = study_pool(spec, config)
            actual = evaluate(model, pool["test"].examples, 2, spec=spec)
            self.assertEqual(without_timing(actual), without_timing(result["test_final"]["in_distribution"]))
            self.assertTrue((directory / "data" / "validation_probes" / "length-8" / "metadata.json").is_file())

    def test_final_and_validation_selected_models_remain_distinct(self):
        def metrics(exact, loss):
            return {"sequence_accuracy": exact, "balanced_accuracy": exact, "token_accuracy": exact,
                    "loss": loss, "example_loss": loss}
        spec = TaskSpec(task="copy", length=4, symbols=8)
        config = replace(self.config(), eval_lengths=())
        with tempfile.TemporaryDirectory() as root:
            with patch("experiments.synthetic_trainers.training.evaluate", side_effect=(
                    metrics(1, 5), metrics(.5, 3), metrics(.8, 1), metrics(.2, 2), metrics(.9, .1))):
                result = train_run(spec, self.model(), config, root)
            self.assertEqual(result["best_step"], 0)
            self.assertEqual(result["test_final"]["in_distribution"]["loss"], 2)
            self.assertEqual(result["test"]["in_distribution"]["loss"], .1)
            final = torch.load(Path(root) / "final.pt", weights_only=True)
            selected = torch.load(Path(root) / "best.pt", weights_only=True)
            self.assertTrue(any(not torch.equal(final[key], selected[key]) for key in final))

    def test_random_control_publishes_code_bits_and_never_calls_accuracy_understanding(self):
        spec = TaskSpec(task="random_lm", length=4, min_length=2, symbols=4, number_limit=16)
        with tempfile.TemporaryDirectory() as root:
            result = train_run(spec, self.model(), replace(self.config(), study="memorization"), root)
            report = result["study"]
            self.assertIn("net_bits_per_parameter", report["final_checkpoint"]["train_compression"])
            self.assertEqual(len(report["final_checkpoint"]["prefix_extraction"]), 3)
            self.assertIsNone(report["generalization_transition"]["generalization_step"])
            self.assertEqual(report["data_scope"], "IID_uniform_payloads")
            self.assertGreater(report["causal_contexts"]["supervised_contexts"], 0)

    def test_sweep_uses_nested_noisy_pools_shared_probes_and_complete_epochs(self):
        spec = TaskSpec(task="copy", length=4, symbols=8)
        config = replace(self.config(), split_policy="disjoint")
        grid = SweepConfig(widths=(8, 16), layer_counts=(1,), sample_sizes=(2, 3), noise_rates=(0, 0.5),
                           model_seeds=(0, 1), data_seeds=(0,), epochs=2)
        with tempfile.TemporaryDirectory() as root:
            report = run_sweep(spec, self.model(), config, grid, root)
            self.assertEqual(len(report["runs"]), 16)
            self.assertEqual(len(report["aggregates"]), 8)
            self.assertTrue((Path(root) / "sweep.csv").is_file())
            validation_hashes, test_hashes = set(), set()
            for row in report["runs"]:
                result = json.loads(Path(row["result"]).read_text())
                self.assertEqual(row["steps"], 2 if row["train_examples"] == 2 else 4)
                self.assertEqual(row["epochs_seen"], 2)
                self.assertEqual(row["budget_mode"], "epochs")
                validation_hashes.add(result["split_fingerprints"]["validation"])
                test_hashes.add(result["split_fingerprints"]["in_distribution"])
                self.assertEqual(result["study"]["validation_probe_fingerprints"], json.loads(Path(report["runs"][0]["result"]).read_text())["study"]["validation_probe_fingerprints"])
            self.assertEqual(len(validation_hashes), 1)
            self.assertEqual(len(test_hashes), 1)
            a = Path(root) / run_name(8, 1, 2, .5, 0, 0) / "data" / "train" / "examples.jsonl"
            b = Path(root) / run_name(16, 1, 3, .5, 0, 1) / "data" / "train" / "examples.jsonl"
            self.assertEqual(a.read_text().splitlines(), b.read_text().splitlines()[:2])

    def test_sweep_keeps_data_seeds_distinct_and_updates_budget_fixed(self):
        spec = TaskSpec(task="copy", length=4, symbols=8)
        grid = SweepConfig(widths=(8,), layer_counts=(1,), sample_sizes=(2, 3), noise_rates=(0,), data_seeds=(0, 4))
        with tempfile.TemporaryDirectory() as root:
            report = run_sweep(spec, self.model(), self.config(), grid, root)
            self.assertEqual({row["steps"] for row in report["runs"]}, {2})
            hashes = {row["data_seed"]: json.loads(Path(row["result"]).read_text())["split_fingerprints"]["validation"] for row in report["runs"]}
            self.assertNotEqual(hashes[0], hashes[4])

    def test_invalid_grid_fails_before_publishing_any_run(self):
        spec = TaskSpec(task="copy", length=4, symbols=8)
        for grid in (SweepConfig(widths=(8, 9), layer_counts=(1,)), SweepConfig(widths=(8,), layer_counts=(1,), noise_rates=(1,))):
            with tempfile.TemporaryDirectory() as root:
                bad_spec = TaskSpec(task="random_lm", length=4) if grid.noise_rates == (1,) else spec
                with self.assertRaises(ValueError):
                    run_sweep(bad_spec, self.model(), self.config(), grid, root)
                self.assertEqual(list(Path(root).iterdir()), [])
        self.assertNotEqual(run_name(8, 1, 2, 0.12345678, 0, 0), run_name(8, 1, 2, 0.12345679, 0, 0))

    def test_bad_prepared_pool_and_profile_controls_are_rejected(self):
        spec = TaskSpec(task="copy", length=4, symbols=8)
        pool = study_pool(spec, self.config())
        bad = {**pool, "train": replace(pool["train"], examples=pool["train"].examples[:2])}
        with tempfile.TemporaryDirectory() as root, self.assertRaises(ValueError):
            train_run(spec, self.model(), self.config(), root, prepared_splits=bad)
        with tempfile.TemporaryDirectory() as root, self.assertRaises(ValueError):
            train_run(spec, self.model(), replace(self.config(), study="standard"), root, prepared_splits=pool)

    def test_summary_does_not_assume_monotone_interpolation_or_universal_capacity(self):
        rows = []
        for size, fitted, loss in ((2, True, 3), (4, False, 1), (6, True, 4), (8, False, 0)):
            rows.append({"width": 8, "layers": 1, "train_examples": size, "label_noise": 0,
                         "seed": 0, "data_seed": 0, "parameters": 100, "final_loss": loss,
                         "selected_loss": loss, "final_error": .2, "selected_error": .1,
                         "final_fit_value": 0 if fitted else .5, "final_fitted": fitted,
                         "training_seconds": 1, "net_gain_bits": size * 2, "bits_per_parameter": size / 50})
        summary = summarize(rows)
        self.assertTrue(summary["observed_fit_frontiers"][0]["nonmonotone_fit"])
        self.assertEqual(summary["observed_fit_frontiers"][0]["largest_observed_fitted_pool"], 6)
        self.assertEqual(summary["capacity_proxies"][0]["maximum_mean_net_gain_bits"], 16)
        sample_curve = next(row for row in summary["curves"] if row["axis"] == "train_examples")
        self.assertIsNotNone(sample_curve["final_loss_witness"])

    def test_public_random_control_and_memorization_sweep_commands(self):
        out = io.StringIO()
        with redirect_stdout(out):
            main(["check", "--trainer", "random_lm", "--length", "4", "--examples", "3"])
        self.assertEqual(json.loads(out.getvalue())["family"], "memorization")
        with tempfile.TemporaryDirectory() as root, redirect_stdout(io.StringIO()):
            main(["sweep", "--study", "memorization", "--trainer", "random_lm", "--length", "4",
                  "--symbols", "4", "--number-limit", "16", "--widths", "8", "--layer-counts", "1",
                  "--heads", "1", "--sample-sizes", "2", "3", "--batch-size", "2", "--steps", "1",
                  "--eval-every", "1", "--validation-examples", "2", "--test-examples", "2",
                  "--eval-lengths", "8", "--output", root])
            report = json.loads((Path(root) / "random_lm" / "sweep.json").read_text())
            self.assertEqual(len(report["runs"]), 2)
            self.assertEqual({row["label_noise"] for row in report["runs"]}, {0})

    def test_cli_rejects_stopping_or_noising_the_random_control(self):
        for flags in (["--stop-at-target"], ["--label-noise", "0.2"], ["--split-policy", "disjoint"]):
            with tempfile.TemporaryDirectory() as root, redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(["train", "--trainer", "random_lm", "--study", "memorization", "--output", root, *flags])
