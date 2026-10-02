"""Archives reproduce actual interpolation peaks and retain training provenance."""

import json
import copy
from pathlib import Path
import shutil
import tempfile
import unittest

from experiments.synthetic_trainers.paper_reproduction.grokking import RunConfig
from experiments.synthetic_trainers.paper_report import export_csv, load_modular, modular_summary, rff_summary, save
from experiments.synthetic_trainers.reproduction import run_campaign


ARCHIVE = Path(__file__).parents[1] / "baselines/fashion_rff_20261002"


class PaperReportTests(unittest.TestCase):
    def test_target_timing_preserves_failed_runs_and_uses_success_support(self):
        measured = json.loads((ARCHIVE.parent / "mod97_fraction50_wd01_reference_20261002/measurements.json").read_text())["runs"][0]
        failed = copy.deepcopy(measured)
        failed["plan"]["config"]["seed"] = 1
        for point in failed["history"]:
            point["heldout"]["accuracy"] = .01
        failed["final"]["heldout"]["accuracy"] = .01
        failed["transition"].update(heldout_onset_step=None, lag_steps=None, delayed_generalization=False)
        report = modular_summary([measured, failed], 2)
        group = report["aggregates"][0]
        self.assertEqual(group["runs"], 2)
        self.assertEqual(group["final_heldout_target_runs"], 1)
        onset = next(point for point in measured["history"] if point["step"] == 31500)
        confirmed = next(point for point in measured["history"] if point["step"] == 31750)
        self.assertEqual(group["heldout_target_onset_training_seconds"],
                         {"mean": onset["training_seconds"], "std": 0, "support": 1})
        self.assertEqual(group["heldout_target_confirmed_wall_seconds"]["mean"], confirmed["wall_seconds"])
        self.assertIsNone(group["phase_diagnostics"][1]["time_to_sustained_heldout_target"])
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "metrics.csv"
            export_csv("modular", report, path)
            self.assertNotIn(b"\r", path.read_bytes())
            import csv
            with path.open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(float(rows[0]["heldout_target_onset_training_seconds"]), onset["training_seconds"])
            self.assertEqual(rows[1]["heldout_target_onset_training_seconds"], "")

    def test_actual_RFF_peak_agrees_with_interpolation_boundary_for_every_seed(self):
        report = json.loads((ARCHIVE / "measurements.json").read_text())
        summary = rff_summary(report)
        self.assertTrue(summary["complete"])
        self.assertEqual(summary["completed_runs"], 123)
        for curve in summary["curves"]:
            self.assertEqual(curve["repeated_full_error_double_descent"], 3)
            self.assertEqual(curve["peak_on_boundary_runs"], 3)
            measured = curve["mean_error_curve"]
            self.assertEqual(measured["peak"]["x"], 1000)
            self.assertGreater(measured["peak_rise"], .5)
            self.assertGreater(measured["second_descent"], .5)
            # Error has both descending branches; MSE does not have a first descent.
            self.assertIsNone(curve["mean_MSE_four_point_witness"])
        self.assertLess(summary["maximum_interpolation_MSE_at_or_above_width"], 1e-10)
        with tempfile.TemporaryDirectory() as root:
            source, archive = Path(root) / "source", Path(root) / "archive"
            source.mkdir()
            for name in ("plan.json", "measurements.json"):
                shutil.copyfile(ARCHIVE / name, source / name)
            save("rff", source, archive)
            self.assertEqual(json.loads((archive / "measurements.json").read_text()), report)
            self.assertEqual(len((archive / "metrics.csv").read_text().splitlines()), 124)
            with self.assertRaises(FileExistsError):
                save("rff", source, archive)
            report["plan"]["status"] = "running"
            (source / "measurements.json").write_text(json.dumps(report))
            with self.assertRaises(ValueError):
                save("rff", source, Path(root) / "incomplete")
            self.assertFalse((Path(root) / "incomplete").exists())

    def test_modular_archive_is_self_contained_and_retains_all_paired_histories(self):
        config = RunConfig(prime=7, train_fraction=.5, width=8, heads=1, layers=1,
                           steps=2, eval_every=1, batch_size=4, device="cpu")
        with tempfile.TemporaryDirectory() as root:
            source, archive = Path(root) / "source", Path(root) / "archive"
            run_campaign(source, config, (0,))
            summary = save("modular", source, archive)
            report = load_modular(archive)
            self.assertTrue(summary["complete"])
            self.assertEqual(report["completed_runs"], 3)
            self.assertEqual(len(json.loads((source / "measurements.json").read_text())["runs"]), 3)
            original = load_modular(source)
            self.assertEqual(report, original)
            for row in report["runs"]:
                self.assertEqual([point["step"] for point in row["history"]], [0, 1, 2])
                self.assertEqual(row["plan"]["corpus"]["split_policy"], "exhaustive_disjoint_two_way_author_protocol")
                self.assertTrue(row["plan"]["source_hashes"])
