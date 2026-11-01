"""Profile exports must derive evidence from the run being reported."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "profile_report", ROOT / "experiments/modular_composition/report_profiles.py")
reporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reporter)


class ModularProfileTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.raw = Path(temporary.name)
        self.patch = patch.object(reporter, "RAW", self.raw)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        for shape in ("wide", "deep"):
            for mode in ("plain", "sampling", "torch"):
                record = {
                    "case": shape, "mode": mode, "source_checkpoint_sha256": "frozen",
                    "recipe": {"batch": 256}, "python": "3.15.2", "torch": "2.15.0", "gpu": "test",
                    "median_updates_per_second": 100 if mode == "plain" else 80,
                    "events": [{"name": "Optimizer.step#ScheduleFreeAdamW.step", "device_total_us": 7},
                               {"name": "complete_schedule_free_update", "calls": 11}],
                }
                self.write(f"{shape}-{mode}.json", record)
            rows = [{"type": "meta", "run_id": shape},
                    {"type": "end", "run_id": shape, "samples_total": 9}]
            (self.raw / f"{shape}.jsonl").write_text("\n".join(map(json.dumps, rows)))
            self.write(f"{shape}-cuda.json", {"traceEvents": [
                {"cat": "kernel", "name": "fmha_cutlass_test", "dur": 60},
                {"cat": "kernel", "name": "other", "dur": 40}]})

    def write(self, name, value):
        (self.raw / name).write_text(json.dumps(value))

    def test_rates_follow_each_retained_console_log(self):
        for shape, rate in (("wide", 3.21), ("deep", 17.89)):
            with self.subTest(shape=shape):
                name = f"{shape}-sampling.log"
                (self.raw / name).write_text(f"Captured 10 samples in 1.00 seconds\nError rate: {rate}\n")
                result = reporter.report(shape)
                self.assertEqual(result["sampler_reported_error_rate"], rate)
                self.assertEqual(result["sampler_reported_error_rate_unit"], "percent")
                self.assertEqual(result["sampler_error_rate_evidence"], {"status": "recorded", "source": name})
                self.assertIn(name, result["artifacts"])
                self.assertEqual(result["torch_profile_updates"], 11)
                self.assertAlmostEqual(result["sampler_wall_overhead_fraction"], .25)
                self.assertAlmostEqual(result["attention_fraction_of_kernel_time"], .6)

    def test_missing_console_log_is_explicitly_unknown(self):
        result = reporter.report("wide")
        self.assertIsNone(result["sampler_reported_error_rate"])
        self.assertEqual(result["sampler_error_rate_evidence"]["status"], "missing_console_log")
        self.assertEqual(result["sampling_valid_samples"], 9)

    def test_recorded_zero_is_distinct_from_missing(self):
        (self.raw / "wide-sampling.log").write_text("Error rate: 0.00\n")
        result = reporter.report("wide")
        self.assertEqual(result["sampler_reported_error_rate"], 0)
        self.assertEqual(result["sampler_error_rate_evidence"]["status"], "recorded")

    def test_incomplete_duplicate_and_invalid_console_statistics_are_rejected(self):
        for contents in ("Captured 10 samples\n", "Error rate: 1\nError rate: 2\n",
                         "Error rate: 100.01\n", "Error rate: nan\n", "Error rate: -1\n"):
            with self.subTest(contents=contents):
                (self.raw / "wide-sampling.log").write_text(contents)
                with self.assertRaisesRegex(ValueError, "exactly one valid"):
                    reporter.report("wide")

    def test_mismatched_profile_recipe_and_checkpoint_are_rejected(self):
        path = self.raw / "wide-sampling.json"
        original = json.loads(path.read_text())
        for key, value in (("recipe", {"batch": 1}), ("source_checkpoint_sha256", "other")):
            with self.subTest(key=key):
                self.write(path.name, original | {key: value})
                with self.assertRaisesRegex(ValueError, f"Mismatched {key}"):
                    reporter.report("wide")

    def test_truncated_and_concatenated_sampler_streams_are_rejected(self):
        for rows in ([{"type": "meta", "run_id": "wide"}],
                     [{"type": "meta", "run_id": "wide"},
                      {"type": "end", "run_id": "another", "samples_total": 9}]):
            with self.subTest(rows=rows):
                (self.raw / "wide.jsonl").write_text("\n".join(map(json.dumps, rows)))
                with self.assertRaisesRegex(ValueError, "one complete JSONL"):
                    reporter.report("wide")
