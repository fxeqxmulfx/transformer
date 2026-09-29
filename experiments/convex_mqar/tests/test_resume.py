"""Resume is numerically identical to uninterrupted deterministic training."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from convex_mqar import engine

from .support import batch, config


class ResumeTests(unittest.TestCase):
    def test_interruption_and_resume_reproduce_model_and_adam_state_exactly(self):
        cfg = config()
        training = batch(1, 67, 8, 16)
        validation = batch(2, 65, 8, 16)
        with tempfile.TemporaryDirectory() as uninterrupted, tempfile.TemporaryDirectory() as resumed:
            with redirect_stdout(io.StringIO()):
                complete, complete_best = engine.train_run(cfg, 8, 8, 0.003, training,
                    validation, uninterrupted, Path(uninterrupted) / "status.json")
                evaluations = 0
                evaluate = engine.evaluate

                def interrupt_on_second_epoch(*args, **kwargs):
                    nonlocal evaluations
                    evaluations += 1
                    if evaluations == 2:
                        raise InterruptedError("Simulated interruption before checkpoint")
                    return evaluate(*args, **kwargs)

                with patch.object(engine, "evaluate", side_effect=interrupt_on_second_epoch):
                    with self.assertRaises(InterruptedError):
                        engine.train_run(cfg, 8, 8, 0.003, training, validation,
                                         resumed, Path(resumed) / "status.json")
                checkpoint = torch.load(Path(resumed) / "current.pt", weights_only=False)
                self.assertEqual(checkpoint["next_epoch"], 1)
                self.assertFalse((Path(resumed) / "result.json").exists())
                restored, restored_best = engine.train_run(cfg, 8, 8, 0.003, training,
                    validation, resumed, Path(resumed) / "status.json")
            for name in ("best_epoch", "epochs_completed", "parameters", "batch_size"):
                self.assertEqual(complete[name], restored[name])
            for name in ("accuracy", "loss", "correct", "query_count"):
                self.assertEqual(complete["validation"][name], restored["validation"][name])
            states = [torch.load(Path(root) / "current.pt", weights_only=False)
                      for root in (uninterrupted, resumed)]
            for name in states[0]["model"]:
                self.assertTrue(torch.equal(states[0]["model"][name], states[1]["model"][name]), name)
            self.assertEqual(states[0]["optimizer"]["param_groups"], states[1]["optimizer"]["param_groups"])
            for parameter, fields in states[0]["optimizer"]["state"].items():
                for name, value in fields.items():
                    self.assertTrue(torch.equal(value, states[1]["optimizer"]["state"][parameter][name]))
            bests = [torch.load(path, weights_only=True) for path in (complete_best, restored_best)]
            for name in bests[0]:
                self.assertTrue(torch.equal(bests[0][name], bests[1][name]), name)
            rows = [json.loads(line) for line in (Path(resumed) / "epochs.jsonl").read_text().splitlines()]
            self.assertEqual([row["epoch"] for row in rows], [1, 2, 3])

    def test_completed_runs_are_reused_without_training_again(self):
        cfg, data = config(epochs=1), batch(count=3, length=8, vocab=16)
        with tempfile.TemporaryDirectory() as root, redirect_stdout(io.StringIO()):
            expected, checkpoint = engine.train_run(cfg, 8, 8, 0.003, data, data,
                                                   root, Path(root) / "status.json")
            before = {p.name: p.read_bytes() for p in Path(root).iterdir() if p.is_file()}
            with patch.object(engine, "RopeTransformer", side_effect=AssertionError("Must reuse result")):
                actual, actual_checkpoint = engine.train_run(cfg, 8, 8, 0.003, None, None,
                                                            root, Path(root) / "status.json")
            after = {p.name: p.read_bytes() for p in Path(root).iterdir() if p.is_file()}
        self.assertEqual(actual, expected)
        self.assertEqual(checkpoint, actual_checkpoint)
        self.assertEqual(before, after)
