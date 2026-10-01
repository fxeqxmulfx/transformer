"""Measurement export preserves separate depth/N ablations and checkpoint scores."""

import json
from pathlib import Path
import tempfile
import unittest

import torch

from experiments.synthetic_trainers.config import ModelSpec, TrainConfig
from experiments.synthetic_trainers.runtime import write_json
from experiments.synthetic_trainers.scaling_report import save, summarize, width_witnesses
from experiments.synthetic_trainers.specs import TaskSpec
from experiments.synthetic_trainers.sweeps import SweepConfig, run_sweep


class ScalingReportTests(unittest.TestCase):
    def test_real_grid_does_not_pool_depth_or_N_and_preserves_archive_provenance(self):
        torch.set_num_threads(1)
        spec = TaskSpec(task="parity", length=4, symbols=2)
        model = ModelSpec(width=8, layers=1, heads=1, init_std=.02)
        config = TrainConfig(steps=1, eval_every=1, batch_size=2, train_examples=2,
                             validation_examples=2, test_examples=2, eval_lengths=(8,),
                             study="double_descent", optimizer="amsgradw", grad_clip=None)
        grid = SweepConfig((8,), (1, 2), (2, 4), (0,), (0, 1), (0,))
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            sweep = run_sweep(spec, model, config, grid, directory / "dd-calibration")
            write_json(directory / "plan.json", {"status": "complete", "phase": "calibration",
                       "dd_training": {"curve_tolerance": .02}})
            report = summarize(directory)
            self.assertTrue(report["complete"])
            self.assertEqual((report["completed_runs"], report["planned_runs"]), (8, 8))
            self.assertEqual({(row["layers"], row["train_examples"]) for row in report["aggregates"]},
                             {(1, 2), (1, 4), (2, 2), (2, 4)})
            for row in report["aggregates"]:
                self.assertEqual(row["runs"], 2)
                self.assertEqual(row["id_test_loss"]["support"], 2)
            self.assertTrue(all(row["all_split_fingerprints_identical"] for row in report["paired_pools"]))
            self.assertIsNone(report["width_witnesses"]["mean"]["loss"])
            target = directory / "archive"
            save(report, directory, target)
            saved = json.loads((target / "measurements.json").read_text())
            for archived, item in zip(saved["runs"], sweep["runs"]):
                measured = json.loads(Path(item["result"]).read_text())
                self.assertEqual(archived["test_final"]["in_distribution"]["example_loss"],
                                 measured["test_final"]["in_distribution"]["example_loss"])
                self.assertEqual(archived["test_selected"]["in_distribution"]["example_loss"],
                                 measured["test"]["in_distribution"]["example_loss"])
                self.assertEqual(archived["provenance"]["source_hashes"], measured["provenance"]["source_hashes"])
                self.assertEqual(archived["split_fingerprints"], measured["split_fingerprints"])
                self.assertEqual([point["step"] for point in archived["history"]], [0, 1])
            with self.assertRaises(FileExistsError):
                save(report, directory, target)
            with self.assertRaises(ValueError):
                save({**report, "complete": False}, directory, directory / "incomplete")
            self.assertFalse((directory / "incomplete").exists())

    def test_width_witness_rejects_mixing_different_depths(self):
        rows = [{"phase": "dd-widths", "width": width, "layers": layers,
                 "train_examples": 512, "label_noise": .2}
                for width, layers in ((16, 2), (32, 6))]
        with self.assertRaisesRegex(ValueError, "fixed depth"):
            width_witnesses([], rows, .02)
