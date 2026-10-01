"""Public commands expose the complete suite and the original reference group."""

from contextlib import redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from experiments.synthetic_trainers.cli import main, task_names
from experiments.synthetic_trainers.specs import TASKS


class CliTests(unittest.TestCase):
    def test_check_runs_all_tasks_and_unique_named_variants(self):
        output = io.StringIO()
        with redirect_stdout(output):
            main(["check", "--examples", "5"])
        rows = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual({row["task"] for row in rows}, set(TASKS))
        self.assertEqual({row["family"] for row in rows}, {"mqar", "lookup", "prefix", "rasp", "rasp_l"})
        self.assertEqual(len(rows), 37)
        self.assertEqual(len({row["variant"] for row in rows}), len(rows))
        self.assertEqual(task_names("prefix", "dyck"), ("dyck",))

    def test_check_rejects_impossible_difficulty(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
            main(["check", "--hops", "8", "--pairs", "8"])
        self.assertEqual(raised.exception.code, 2)

    def test_real_cli_training_runs_every_mode_and_writes_reports(self):
        with tempfile.TemporaryDirectory() as root:
            command = [sys.executable, "-m", "experiments.synthetic_trainers", "train",
                       "--trainer", "core", "--output", root, "--steps", "1", "--eval-every", "1",
                       "--length", "24", "--min-length", "22", "--pairs", "4", "--queries", "2",
                       "--width", "8", "--layers", "1", "--heads", "1", "--batch-size", "2",
                       "--train-examples", "4", "--validation-examples", "4", "--test-examples", "4",
                       "--eval-lengths", "48"]
            result = subprocess.run(command, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr)
            for task in ("mqar", "lookup", "dyck", "blocks"):
                report = json.loads((Path(root) / task / "seed-0" / "result.json").read_text())
                self.assertEqual(report["task"], task)
                self.assertEqual(report["steps_completed"], 1)
                self.assertIn("length-48", report["test"])

    def test_cli_single_generated_variant_preserves_supplied_format_and_summary(self):
        with tempfile.TemporaryDirectory() as root:
            command = [sys.executable, "-m", "experiments.synthetic_trainers", "train",
                       "--trainer", "addition", "--variants", "core", "--output", root,
                       "--addition-order", "reverse", "--index-hints", "--carry-length", "3",
                       "--length", "4", "--symbols", "8", "--number-limit", "16",
                       "--steps", "1", "--eval-every", "1", "--width", "8", "--layers", "1",
                       "--heads", "1", "--batch-size", "2", "--train-examples", "2",
                       "--validation-examples", "2", "--test-examples", "2", "--eval-lengths", "8"]
            observed = subprocess.run(command, capture_output=True, text=True, timeout=60)
            self.assertEqual(observed.returncode, 0, observed.stderr)
            rows = json.loads((Path(root) / "suite.json").read_text())
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["variant"], "addition-reverse-hints-carry-3")
            report = json.loads(Path(rows[0]["result"]).read_text())
            self.assertEqual(report["provenance"]["task"]["carry_length"], 3)
            self.assertEqual(report["provenance"]["task"]["addition_order"], "reverse")
            self.assertIn("teacher_forced", report["test"]["in_distribution"])
            self.assertIn("hard-carry-length-8", report["test"])

    def test_cli_preflights_ood_alphabet_before_creating_any_run(self):
        with tempfile.TemporaryDirectory() as root:
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
                main(["train", "--trainer", "copy", "--variants", "core", "--unique",
                      "--symbols", "4", "--length", "4", "--eval-lengths", "8", "--output", root])
            self.assertEqual(raised.exception.code, 2)
            self.assertEqual(list(Path(root).iterdir()), [])
