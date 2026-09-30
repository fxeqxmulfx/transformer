"""Threshold timing integrity, coarse observations, and validation-only selection."""

import csv
import io
import json
from pathlib import Path
import tempfile
import unittest

from convex_mqar.milestones import THRESHOLDS, summarize_epochs
from convex_mqar.validation_milestones import build_report, export_report


def epoch(number, accuracy, seconds, compiled=True):
    return {"epoch": number, "length": 64, "width": 64, "learning_rate": .001,
            "training_seconds": seconds, "compiled_loss": compiled,
            "validation": {"accuracy": accuracy, "loss": 1 - accuracy,
                           "correct": round(accuracy * 10_000), "query_count": 10_000,
                           "seconds": float(number)}}


def fixtures(root):
    config = {"lengths": [64], "widths": [64], "learning_rates": [.001, .003],
              "epochs": 2, "seed": 0}
    for family in ("softmax", "sparsemax"):
        directory = Path(root) / family
        directory.mkdir()
        metadata = {"config": config} if family == "sparsemax" else config
        (directory / "config.json").write_text(json.dumps(metadata))
        for rate in config["learning_rates"]:
            run = directory / f"n64-d64-lr{rate:.8g}"
            run.mkdir()
            # The quickest crossing is not the run with best validation.
            rows = [epoch(1, .1, 50), epoch(2, .995, 100)] if rate == .001 else [
                epoch(1, .99, 10), epoch(2, .8, 20)]
            for row in rows:
                row["learning_rate"] = rate
            (run / "epochs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            (run / "result.json").write_text("{}\n")
        # A malformed test file must not affect validation-only selection.
        (directory / "comparison.json").write_text("test data must never be read")
    return Path(root) / "softmax", Path(root) / "sparsemax"


class MilestoneTests(unittest.TestCase):
    def test_first_inclusive_crossing_with_real_training_and_validation_timers(self):
        accuracies = [.49, .50, .76, .90, .96, .99, .20]
        times = [10, 22, 35, 52, 70, 85, 100]
        rows = [epoch(i + 1, a, t, compiled=i > 0) for i, (a, t) in
                enumerate(zip(accuracies, times))]
        hits = summarize_epochs(rows)["milestones"]
        for target, number, seconds in zip(THRESHOLDS, (2, 3, 4, 5, 6), (22, 35, 52, 70, 85)):
            self.assertEqual(hits[str(target)]["epoch"], number)
            self.assertEqual(hits[str(target)]["training_seconds"], seconds)
            self.assertEqual(hits[str(target)]["training_plus_validation_seconds"],
                             seconds + number * (number + 1) / 2)
            self.assertFalse(hits[str(target)]["sustained_to_end"])
        self.assertEqual(hits["50"]["execution_modes_until_crossing"], ["compiled", "eager"])

    def test_large_epoch_jump_has_equal_observation_times_without_interpolation(self):
        hits = summarize_epochs([epoch(1, .1, 9), epoch(2, .995, 22)])["milestones"]
        for hit in hits.values():
            self.assertEqual(hit["epoch"], 2)
            self.assertEqual(hit["training_seconds"], 22)
            self.assertEqual(hit["previous_validation_epoch"], 1)
            self.assertEqual(hit["previous_training_seconds"], 9)
            self.assertTrue(hit["sustained_to_end"])

    def test_unreached_thresholds_are_null_instead_of_zero_or_the_final_time(self):
        summary = summarize_epochs([epoch(1, .05, 9), epoch(2, .25, 22)])
        self.assertTrue(all(hit is None for hit in summary["milestones"].values()))
        self.assertEqual(summary["total_training_seconds"], 22)

    def test_missing_validation_timer_is_not_fabricated(self):
        rows = [epoch(1, .1, 9), epoch(2, .99, 22)]
        del rows[0]["validation"]["seconds"]
        self.assertIsNone(summarize_epochs(rows)["milestones"]["99"]["training_plus_validation_seconds"])

    def test_retry_gaps_invalid_counts_and_decreasing_time_are_rejected(self):
        cases = ([epoch(1, .1, 10), epoch(1, .99, 20)],
                 [epoch(1, .1, 10), epoch(3, .99, 20)],
                 [epoch(1, .1, 20), epoch(2, .99, 10)],
                 [epoch(1, .1, float("nan"))])
        for rows in cases:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                summarize_epochs(rows)
        row = epoch(1, .1, 10)
        row["validation"]["correct"] = 5
        with self.assertRaisesRegex(ValueError, "query counts"):
            summarize_epochs([row])

    def test_report_selects_validation_quality_without_reading_test_results(self):
        with tempfile.TemporaryDirectory() as root:
            softmax, sparsemax = fixtures(root)
            report = build_report(softmax, sparsemax, lengths=(64,), selection="best")
            self.assertEqual(len(report["all_runs"]), 4)
            self.assertEqual(len(report["selected_runs"]), 2)
            for run in report["selected_runs"]:
                self.assertEqual(run["learning_rate"], .001)
                self.assertEqual(run["milestones"]["99"]["training_seconds"], 100)
            output = Path(root) / "report"
            export_report(report, output)
            self.assertEqual(json.loads(output.with_suffix(".json").read_text()), report)
            selected_rows = list(csv.DictReader(io.StringIO(output.with_suffix(".csv").read_text())))
            all_rows = list(csv.DictReader(io.StringIO(output.with_name("report_all_lr.csv").read_text())))
            self.assertEqual(len(selected_rows), 10)
            self.assertEqual(len(all_rows), 20)

    def test_first99_policy_uses_first_crossing_even_if_accuracy_later_collapses(self):
        with tempfile.TemporaryDirectory() as root:
            softmax, sparsemax = fixtures(root)
            report = build_report(softmax, sparsemax, lengths=(64,), selection="first99")
            self.assertEqual(report["selection_policy"], "first99")
            for run in report["selected_runs"]:
                self.assertEqual(run["learning_rate"], .003)
                self.assertEqual(run["milestones"]["99"]["epoch"], 1)
                self.assertFalse(run["milestones"]["99"]["sustained_to_end"])

    def test_first_99_percent_prioritizes_epoch_then_measured_seconds(self):
        with tempfile.TemporaryDirectory() as root:
            softmax, sparsemax = fixtures(root)
            for directory in (softmax, sparsemax):
                path = directory / "n64-d64-lr0.001" / "epochs.jsonl"
                rows = [epoch(1, .99, 50), epoch(2, .999, 100)]
                path.write_text("".join(json.dumps(r) + "\n" for r in rows))
                path = directory / "n64-d64-lr0.003" / "epochs.jsonl"
                rows = [epoch(1, .98, 1), epoch(2, .99, 2)]
                for row in rows:
                    row["learning_rate"] = .003
                path.write_text("".join(json.dumps(r) + "\n" for r in rows))
            report = build_report(softmax, sparsemax, lengths=(64,))
            self.assertTrue(all(r["learning_rate"] == .001 for r in report["selected_runs"]))
            # Now both candidates first reach 99% in epoch 1; the faster one wins.
            for directory in (softmax, sparsemax):
                path = directory / "n64-d64-lr0.003" / "epochs.jsonl"
                rows = [epoch(1, .99, 1), epoch(2, .99, 2)]
                for row in rows:
                    row["learning_rate"] = .003
                path.write_text("".join(json.dumps(r) + "\n" for r in rows))
            report = build_report(softmax, sparsemax, lengths=(64,))
            self.assertTrue(all(r["learning_rate"] == .003 for r in report["selected_runs"]))

    def test_no_99_percent_crossing_has_no_selected_lr_and_explicit_csv_status(self):
        with tempfile.TemporaryDirectory() as root:
            softmax, sparsemax = fixtures(root)
            for directory in (softmax, sparsemax):
                for path in directory.glob("*/epochs.jsonl"):
                    rows = [json.loads(line) for line in path.read_text().splitlines()]
                    for row in rows:
                        row["validation"].update(accuracy=.25, correct=2500, loss=.75)
                    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
            report = build_report(softmax, sparsemax, lengths=(64,))
            self.assertEqual(report["selected_runs"], [])
            self.assertEqual(len(report["unselected_groups"]), 2)
            output = Path(root) / "report"
            export_report(report, output)
            rows = list(csv.DictReader(io.StringIO(output.with_suffix(".csv").read_text())))
            self.assertEqual(len(rows), 10)
            self.assertTrue(all(r["learning_rate"] == "" and r["selection_status"] ==
                                "99_percent_unreached" for r in rows))

    def test_different_configs_and_incomplete_sweeps_are_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            softmax, sparsemax = fixtures(root)
            path = sparsemax / "config.json"
            contents = json.loads(path.read_text())
            contents["config"]["seed"] = 11
            path.write_text(json.dumps(contents))
            with self.assertRaisesRegex(ValueError, "different training configurations"):
                build_report(softmax, sparsemax, lengths=(64,))
            contents["config"]["seed"] = 0
            path.write_text(json.dumps(contents))
            next(sparsemax.glob("*/result.json")).unlink()
            with self.assertRaisesRegex(ValueError, "not complete"):
                build_report(softmax, sparsemax, lengths=(64,))


if __name__ == "__main__":
    unittest.main()
