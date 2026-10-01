"""Saved measurements aggregate without selecting models on test scores."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import torch

from experiments.synthetic_trainers.baseline_report import archive, moments, summarize
from experiments.synthetic_trainers.config import ModelSpec, TrainConfig
from experiments.synthetic_trainers.runtime import write_json
from experiments.synthetic_trainers.specs import TaskSpec
from experiments.synthetic_trainers.training import train_run


class BaselineReportTests(unittest.TestCase):
    def test_missing_event_support_is_not_imputed_as_zero(self):
        self.assertIsNone(moments([None, None]))
        self.assertEqual(moments([None, 3]), {"mean": 3, "std": 0, "support": 1})
        report = moments([1, 3])
        self.assertAlmostEqual(report["std"], 2 ** .5)

    def test_two_real_checkpoints_preserve_final_selected_and_seed_support(self):
        torch.set_num_threads(1)
        spec = TaskSpec(task="parity", length=4, symbols=4)
        model = ModelSpec(width=8, layers=1, heads=1, init_std=.02)
        config = TrainConfig(steps=2, eval_every=1, batch_size=2, train_examples=4,
                             validation_examples=2, test_examples=2, eval_lengths=(8,),
                             study="double_descent", split_policy="disjoint", optimizer="amsgradw", grad_clip=None)
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            index = []
            expected = []
            for seed in (0, 1):
                name = f"transitions/parity-none/width-8/noise-0/seed-{seed}"
                result = train_run(spec, model, replace(config, seed=seed), directory / name)
                index.append({"name": name, "result": f"{name}/result.json"})
                expected.append(result)
            write_json(directory / "runs.json", index)
            write_json(directory / "manifest.json", {"status": "complete", "planned_runs": 2,
                       "gpu": None, "gpu_memory_bytes": None, "device": "cpu", "torch": torch.__version__,
                       "cuda_runtime": None, "cpu_threads": 1})
            summary = summarize(directory)
            self.assertEqual(summary["completed_runs"], 2)
            aggregate, = summary["aggregates"]
            self.assertEqual(aggregate["model_seeds"], [0, 1])
            self.assertEqual(aggregate["test_id_loss"]["support"], 2)
            self.assertAlmostEqual(aggregate["test_id_loss"]["mean"],
                                   sum(row["test_final"]["in_distribution"]["example_loss"] for row in expected) / 2)
            for row, report in zip(summary["runs"], expected):
                self.assertEqual(row["test_selected"], report["test"])
                self.assertEqual(row["test_final"], report["test_final"])
            self.assertIsNone(summary["capacity_mean_witnesses"]["test_id_loss"])
            (directory / "metrics.csv").write_text("name,test_id_accuracy\n")
            archive(summary, directory, directory / "archive")
            saved = json.loads((directory / "archive" / "measurements.json").read_text())
            self.assertEqual(saved["runs"][0]["test_final"]["in_distribution"]["sequence_accuracy"],
                             expected[0]["test_final"]["in_distribution"]["sequence_accuracy"])
            self.assertEqual([point["step"] for point in saved["runs"][0]["history"]], [0, 1, 2])
            self.assertEqual(saved["runs"][0]["split_fingerprints"], expected[0]["split_fingerprints"])
            self.assertEqual(saved["runs"][0]["source_hashes"], expected[0]["provenance"]["source_hashes"])
