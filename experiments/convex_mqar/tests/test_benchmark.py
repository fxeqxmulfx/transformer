"""Common-test isolation, selection, directory locks, and the complete CPU path."""

from contextlib import redirect_stdout
import fcntl
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from convex_mqar import benchmark
from convex_mqar.convex import ConvexRecall
from convex_mqar.rope import RopeTransformer

from .support import config


class BenchmarkTests(unittest.TestCase):
    def test_real_cpu_pipeline_uses_one_common_test_after_all_candidates(self):
        cfg = config(epochs=2)
        calls = []
        train, evaluate = benchmark.train_run, benchmark.evaluate

        def observed_training(config, length, width, rate, training, validation, *args):
            self.assertEqual(len(training[0]), cfg.train_examples)
            self.assertEqual(len(validation[0]), cfg.validation_examples)
            calls.append(("train", rate))
            return train(config, length, width, rate, training, validation, *args)

        def observed_evaluation(model, data, *args, **kwargs):
            self.assertEqual(len(data[0]), cfg.test_examples)
            calls.append(("test", id(data), isinstance(model, ConvexRecall)))
            return evaluate(model, data, *args, **kwargs)

        with tempfile.TemporaryDirectory() as root, redirect_stdout(io.StringIO()):
            output = Path(root) / "output"
            with patch.object(benchmark.torch, "set_num_threads"), \
                    patch.object(benchmark, "train_run", side_effect=observed_training), \
                    patch.object(benchmark, "evaluate", side_effect=observed_evaluation):
                report = benchmark.run(cfg, output, Path(root) / "data")
            self.assertEqual([item[0] for item in calls], ["train", "train", "test", "test"])
            self.assertEqual(calls[-2][1], calls[-1][1])
            row = report["results"][0]
            self.assertEqual(row["convex"]["accuracy"], 1.)
            self.assertEqual(row["rope"]["query_count"], 38)
            self.assertEqual(row["convex"]["query_count"], 38)
            self.assertEqual(len(set(row["splits"].values())), 3)
            self.assertTrue(0 <= row["rope"]["accuracy"] <= 1)
            self.assertEqual(json.loads((output / "status.json").read_text())["event"], "complete")
            self.assertEqual(json.loads((output / "comparison.json").read_text()), report)
            candidates = [json.loads(path.read_text()) for path in output.glob("*/result.json")]
            expected = max(candidates, key=lambda r: (r["validation"]["accuracy"], -r["validation"]["loss"]))
            self.assertEqual(row["selected_run"], expected)

    def test_validation_selection_loads_the_right_checkpoint_before_test(self):
        cfg = config(learning_rates=(0.003, 0.01, 0.02))
        scores = {0.003: (0.6, 2.), 0.01: (0.6, 1.), 0.02: (0.5, 0.1)}
        calls = []

        def candidate(config, length, width, rate, training, validation, directory, status):
            self.assertEqual(len(training[0]), cfg.train_examples)
            self.assertEqual(len(validation[0]), cfg.validation_examples)
            calls.append(("train", rate))
            model = RopeTransformer(cfg.vocab, width)
            with torch.no_grad():
                model.embedding.weight.fill_(rate)
            Path(directory).mkdir()
            checkpoint = Path(directory) / "best.pt"
            torch.save(model.state_dict(), checkpoint)
            accuracy, loss = scores[rate]
            return {"learning_rate": rate, "validation": {"accuracy": accuracy, "loss": loss}}, checkpoint

        def test_evaluation(model, data, size, config, convex=False):
            self.assertEqual(len(data[0]), cfg.test_examples)
            self.assertEqual(len(calls), 3 if not convex else 4)
            if not convex:
                torch.testing.assert_close(model.embedding.weight,
                                           torch.full_like(model.embedding.weight, 0.01))
            calls.append(("test", convex))
            return {"accuracy": 1. if convex else 0.75}

        with tempfile.TemporaryDirectory() as root, redirect_stdout(io.StringIO()):
            with patch.object(benchmark.torch, "set_num_threads"), \
                    patch.object(benchmark, "train_run", side_effect=candidate), \
                    patch.object(benchmark, "evaluate", side_effect=test_evaluation):
                report = benchmark.run(cfg, Path(root) / "output", Path(root) / "data")
        self.assertEqual(report["results"][0]["selected_run"]["learning_rate"], 0.01)
        self.assertEqual(report["results"][0]["accuracy_difference"], 0.25)

    def test_concurrent_process_cannot_share_the_run_directory(self):
        with tempfile.TemporaryDirectory() as root:
            with (Path(root) / ".lock").open("w") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with patch.object(benchmark, "run_locked") as inner:
                    with self.assertRaisesRegex(RuntimeError, "Another process"):
                        benchmark.run(config(), root, Path(root) / "data")
                    inner.assert_not_called()

    def test_incompatible_configuration_cannot_reuse_checkpoints(self):
        with tempfile.TemporaryDirectory() as root:
            contents = config().to_dict()
            contents["seed"] = 999
            (Path(root) / "config.json").write_text(json.dumps(contents))
            with patch.object(benchmark.torch, "set_num_threads"), \
                    patch.object(benchmark, "train_run") as training:
                with self.assertRaisesRegex(RuntimeError, "different configuration"):
                    benchmark.run(config(), root, Path(root) / "data")
                training.assert_not_called()

    def test_cuda_request_does_not_fall_back_to_cpu(self):
        with tempfile.TemporaryDirectory() as root:
            with patch.object(benchmark.torch, "set_num_threads"), \
                    patch.object(benchmark.torch.cuda, "is_available", return_value=False), \
                    patch.object(benchmark, "train_run") as training:
                with self.assertRaisesRegex(RuntimeError, "CUDA is required"):
                    benchmark.run(config(device="cuda"), root, Path(root) / "data")
                training.assert_not_called()
