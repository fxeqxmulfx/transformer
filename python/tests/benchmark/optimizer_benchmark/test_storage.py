from pathlib import Path
import tempfile
import unittest

from gpt_mini.infrastructure.benchmark.optimizer_benchmark.storage import read_results, summarize, winners, write_json


class StorageTests(unittest.TestCase):
    def row(self, name, seed, test_loss, *, status="ok"):
        return {"phase": "final", "attention": "softmax", "method": name, "seed": seed,
                "lr": 0.001, "status": status, "test_loss": test_loss,
                "validation_loss": 3.0, "train_seconds": 10.0, "optimizer_ms": 2.0,
                "peak_memory_mib": 100.0, "guard_acceptance": None}

    def test_only_complete_three_seed_methods_can_win(self):
        rows = [self.row("complete", s, 2 + s * 0.1) for s in [0, 1, 2]]
        rows += [self.row("partial", 0, 0.1), self.row("partial", 1, None, status="failed")]
        rows.append({"phase": "screen", "attention": "softmax", "method": "screen_only"})
        summary = summarize(rows, [0, 1, 2])
        self.assertEqual(winners(summary)["softmax"]["method"], "complete")
        self.assertEqual(summary[0]["successful_seeds"], 3)
        self.assertAlmostEqual(summary[0]["test_loss_mean"], 2.1)
        self.assertAlmostEqual(summary[0]["test_loss_std"], 0.1)

    def test_json_roundtrip_and_nonfinite_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            write_json(path, {"value": 1})
            self.assertIn('"value": 1', path.read_text())
            with self.assertRaises(ValueError):
                write_json(path, {"value": float("nan")})
            self.assertIn('"value": 1', path.read_text())
            raw = Path(directory) / "runs.jsonl"
            raw.write_text('{"method":"amsgrad"}\n')
            self.assertEqual(read_results(raw), [{"method": "amsgrad"}])
