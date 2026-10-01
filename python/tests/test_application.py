"""Exercise orchestration, resume, and integrity with a fake training port."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from gpt_mini.application.benchmark import RunBenchmark
from gpt_mini.domain.benchmark import Request
from gpt_mini.infrastructure.results import FilesystemResults


class FakeTraining:
    def __init__(self, fail=False):
        self.runs, self.checked = [], []
        self.fail = fail
        self.configured = False

    def preflight(self):
        if self.fail:
            raise RuntimeError("Contract failed")
        return {"tests": 1, "failures": 0, "errors": 0, "skipped": 0}

    def dataset_info(self):
        return {"sha256": "fixed", "boundaries": [90, 95, 100], "characters": "ab"}

    def environment(self):
        return {"backend": "fake"}

    def configure(self, model, batch, stopping):
        self.configured = True

    def run(self, job, artifacts):
        self.runs.append(job.identifier)
        return {"id": job.identifier, "attention": job.attention, "method": job.method,
                "seed": job.seed, "lr": job.rate, "status": "ok", "test_loss": 1.5,
                "validation_loss": 1.4, "actual_steps": 2, "best_step": 2,
                "train_seconds": 1., "total_seconds": 2., "optimizer_ms": .1,
                "peak_memory_mib": 1., "stop_reason": "max_steps"}

    def validate(self, row):
        self.checked.append(row["id"])
        return {"id": row["id"], "test_loss": row["test_loss"]}


class ApplicationTests(unittest.TestCase):
    request = Request(methods=("adamw", "amsgradw"), attentions=("softmax",), seeds=(0,))

    def test_persistence_resume_and_checkpoint_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            training = FakeTraining()
            result = RunBenchmark(training, FilesystemResults(directory)).execute(self.request)
            self.assertTrue(result["complete"])
            self.assertEqual(len(training.runs), 2)
            resumed = FakeTraining()
            RunBenchmark(resumed, FilesystemResults(directory)).execute(self.request)
            self.assertEqual(resumed.runs, [])
            self.assertEqual(set(resumed.checked), set(training.runs))
            self.assertIn("amsgradw", (Path(directory) / "REPORT.md").read_text())

    def test_tests_only_has_no_training_or_dataset_setup(self):
        with tempfile.TemporaryDirectory() as directory:
            training = FakeTraining()
            RunBenchmark(training, FilesystemResults(directory)).execute(replace(self.request, tests_only=True))
            self.assertFalse(training.configured)
            self.assertEqual(training.runs, [])
            self.assertFalse((Path(directory) / "metadata.json").exists())

    def test_failed_contract_prevents_training(self):
        with tempfile.TemporaryDirectory() as directory:
            training = FakeTraining(fail=True)
            with self.assertRaisesRegex(RuntimeError, "Contract"):
                RunBenchmark(training, FilesystemResults(directory)).execute(self.request)
            self.assertEqual(training.runs, [])
            self.assertFalse((Path(directory) / "metadata.json").exists())

    def test_changed_settings_cannot_append_to_old_results(self):
        with tempfile.TemporaryDirectory() as directory:
            RunBenchmark(FakeTraining(), FilesystemResults(directory)).execute(self.request)
            training = FakeTraining()
            with self.assertRaisesRegex(ValueError, "settings"):
                RunBenchmark(training, FilesystemResults(directory)).execute(replace(self.request, batch=16))
            self.assertEqual(training.runs, [])

    def test_validate_only_reuses_saved_plan_and_has_no_updates(self):
        with tempfile.TemporaryDirectory() as directory:
            RunBenchmark(FakeTraining(), FilesystemResults(directory)).execute(self.request)
            training = FakeTraining()
            result = RunBenchmark(training, FilesystemResults(directory)).execute(replace(self.request, validate_only=True))
            self.assertTrue(result["complete"])
            self.assertEqual(training.runs, [])
            self.assertEqual(len(training.checked), 2)

    def test_duplicate_raw_records_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            RunBenchmark(FakeTraining(), FilesystemResults(directory)).execute(self.request)
            raw = Path(directory) / "runs.jsonl"
            with raw.open("a") as stream:
                stream.write(raw.read_text().splitlines()[0] + "\n")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                RunBenchmark(FakeTraining(), FilesystemResults(directory)).execute(self.request)

    def test_changed_rate_in_a_saved_row_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            RunBenchmark(FakeTraining(), FilesystemResults(directory)).execute(self.request)
            raw = Path(directory) / "runs.jsonl"
            rows = [json.loads(line) for line in raw.read_text().splitlines()]
            rows[0]["lr"] *= 2
            raw.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            with self.assertRaisesRegex(ValueError, "paired plan"):
                RunBenchmark(FakeTraining(), FilesystemResults(directory)).execute(self.request)
