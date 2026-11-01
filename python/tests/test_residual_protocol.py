"""The task145 wall ceiling cannot silently reset or turn censorship into success."""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from lab.dsl import describe
from lab.infrastructure.engine.residual import ResidualEagerStepper
from lab.infrastructure.store import RunDirectory
from test_residual_transformer import recipe

HERE = Path(__file__).resolve().parents[2] / "experiments/residual_generalization"


def module(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


phase = module("phase")
runner = module("run_case")


class ResidualProtocolTests(unittest.TestCase):
    def test_durable_phase_refuses_a_second_budget_and_expires(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(phase, "LEDGER", Path(directory) / "phase.json"):
            phase.start()
            self.assertLessEqual(phase.require_budget(), 7200)
            with self.assertRaises(FileExistsError):
                phase.start()
            record = json.loads(phase.LEDGER.read_text())
            record["deadline_monotonic"] = 0
            phase.write(phase.LEDGER, record)
            with self.assertRaises(TimeoutError):
                phase.require_budget()

    def test_expired_case_keeps_current_state_and_is_not_marked_finished(self):
        spec = recipe()
        with tempfile.TemporaryDirectory() as directory, patch.object(runner, "require_budget", return_value=100):
            run = RunDirectory(directory)
            run.begin(describe(spec), {}, "unit test")
            training = runner.BoundedTraining(spec, run, lambda row: None, ResidualEagerStepper, allowance=0)
            training.save(0)
            result = training()
            self.assertEqual(result["stop"], {"step": 0, "reason": "wall_budget_censored"})
            self.assertEqual(result["updates"], 0)
            self.assertEqual(result["requested_updates"], spec.budget.updates)
            self.assertEqual(run.checkpoint("cpu")["step"], 0)
            self.assertNotIn("test", result["final"])


if __name__ == "__main__":
    unittest.main()
