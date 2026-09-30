"""Exclude every post-crossing validation drop when choosing a learning rate."""

import csv
import io
import json
from pathlib import Path
import tempfile
import unittest

from convex_mqar.validation_milestones import build_report, export_report

from .test_milestones import epoch, fixtures


class StableSelectionTests(unittest.TestCase):
    def test_an_excluded_active_length_cannot_break_the_completed_report(self):
        with tempfile.TemporaryDirectory() as root:
            softmax, sparsemax = fixtures(root)
            active = sparsemax / "n512-d64-lr0.001"
            active.mkdir()
            (active / "epochs.jsonl").write_text('{"event": "epoch",')
            report = build_report(softmax, sparsemax, lengths=(64,))
            self.assertEqual(len(report["selected_runs"]), 2)

    def test_default_rejects_the_faster_candidate_that_later_drops(self):
        with tempfile.TemporaryDirectory() as root:
            softmax, sparsemax = fixtures(root)
            report = build_report(softmax, sparsemax, lengths=(64,))
            self.assertEqual(report["selection_policy"], "stable99")
            for run in report["selected_runs"]:
                self.assertEqual(run["learning_rate"], .001)
                self.assertEqual(run["milestones"]["99"]["epoch"], 2)
                self.assertTrue(run["milestones"]["99"]["sustained_to_end"])

    def test_every_epoch_is_checked_and_exactly_99_percent_qualifies(self):
        with tempfile.TemporaryDirectory() as root:
            softmax, sparsemax = fixtures(root)
            for directory in (softmax, sparsemax):
                config_path = directory / "config.json"
                metadata = json.loads(config_path.read_text())
                metadata.get("config", metadata)["epochs"] = 3
                config_path.write_text(json.dumps(metadata))
                stable = [epoch(1, .99, 50), epoch(2, .99, 100), epoch(3, .99, 150)]
                # Final accuracy recovers above 99%; the intervening drop
                # still rejects this run under the user's strict policy.
                recovering = [epoch(1, .995, 10), epoch(2, .9899, 20), epoch(3, .999, 30)]
                for rate, rows in ((.001, stable), (.003, recovering)):
                    for row in rows:
                        row["learning_rate"] = rate
                    path = directory / f"n64-d64-lr{rate:.8g}" / "epochs.jsonl"
                    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
            report = build_report(softmax, sparsemax, lengths=(64,))
            self.assertTrue(all(r["learning_rate"] == .001 for r in report["selected_runs"]))

    def test_reached_then_dropped_is_distinct_from_never_reached(self):
        with tempfile.TemporaryDirectory() as root:
            softmax, sparsemax = fixtures(root)
            for directory in (softmax, sparsemax):
                for path in directory.glob("*/epochs.jsonl"):
                    rate = json.loads(path.read_text().splitlines()[0])["learning_rate"]
                    rows = [epoch(1, .99, 10), epoch(2, .8, 20)]
                    for row in rows:
                        row["learning_rate"] = rate
                    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
            report = build_report(softmax, sparsemax, lengths=(64,))
            self.assertEqual(report["selected_runs"], [])
            self.assertEqual(len(report["unselected_groups"]), 2)
            self.assertTrue(all(g["selection_status"] == "99_percent_not_sustained"
                                for g in report["unselected_groups"]))
            output = Path(root) / "report"
            export_report(report, output)
            rows = list(csv.DictReader(io.StringIO(output.with_suffix(".csv").read_text())))
            self.assertTrue(all(r["selection_status"] == "99_percent_not_sustained" and
                                r["learning_rate"] == "" for r in rows))


if __name__ == "__main__":
    unittest.main()
