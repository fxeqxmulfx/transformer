"""First99% is validation-only, checkpointed, resumable and independent of test."""

import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from lab.dsl import AdaptiveResidual, describe, swap
from lab.infrastructure.engine.residual import ResidualEagerStepper
from lab.infrastructure.store import RunDirectory
from test_residual_transformer import recipe

PATH = Path(__file__).resolve().parents[2] / "experiments/adaptive_residual/protocol.py"
SPEC = importlib.util.spec_from_file_location("adaptive_protocol", PATH)
protocol = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(protocol)


class AdaptiveProtocolTests(unittest.TestCase):
    def test_first99_validation_stops_without_training_threshold_or_test_access(self):
        spec = swap(recipe(updates=20), "model", AdaptiveResidual(8, 2, 2, 8))
        calls = []

        def score(self, split):
            calls.append(split)
            if split == "train":
                return {"accuracy": .5, "loss": 1.}
            if split != "validation":
                raise AssertionError("Final test opened during learning")
            return {"accuracy": .98 if len(calls) == 2 else .99, "loss": .1}

        with tempfile.TemporaryDirectory() as directory, patch.object(ResidualEagerStepper, "evaluate", score):
            run = RunDirectory(directory)
            run.begin(describe(spec), {}, "first99 test")
            training = protocol.CrossingTraining(spec, run, lambda row: None, ResidualEagerStepper)
            result = training()
            self.assertEqual(result["stop"], {"step": 2, "reason": "validation99"})
            checkpoint = run.checkpoint("cpu")
            again = protocol.CrossingTraining(spec, run, lambda row: None, ResidualEagerStepper)()
            self.assertEqual(again["stop"], result["stop"])
            torch.testing.assert_close(run.checkpoint("cpu")["model"], checkpoint["model"], rtol=0, atol=0)
            self.assertEqual(calls, ["train", "validation", "train", "validation"])
            self.assertTrue(all("test" not in row for row in run.records("history")))


if __name__ == "__main__":
    unittest.main()
