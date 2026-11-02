"""Matched-control chunk decisions preserve trajectories and never turn censoring into success."""

import copy
import importlib.util
import json
from pathlib import Path
import platform
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from lab.dsl import ResidualMaxAutotune, swap
from lab.infrastructure.benchmarks import build_task
from lab.infrastructure.engine.residual import ResidualCompiledStepper
from test_residual_basis import tiny
from test_residual_transformer import train

path = Path(__file__).resolve().parents[2] / "scripts/basis_continuation.py"
spec = importlib.util.spec_from_file_location("control_continuation", path)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class ControlContinuationTests(unittest.TestCase):
    def test_decisions_cannot_drop_cases_reopen_crossings_or_change_endpoint(self):
        record = {"status": "budget", "stop": {"step": 10}, "checkpoint_sha256": "state"}
        phase = {"status": "awaiting_continuation", "cases": {"case": record}}
        declaration = {"cases": [{"label": "case"}]}
        plan = {"label": "case", "step": 10, "checkpoint_sha256": "state",
                "action": "extend", "updates": 20, "reason": "Validation is improving"}
        decision = {"protocol_sha256": "protocol", "decisions": [plan]}
        runner.validate_decisions(phase, declaration, decision, "protocol")
        for key, value in (("step", 9), ("checkpoint_sha256", "changed"), ("updates", 10),
                           ("updates", True), ("reason", ""), ("action", "censor")):
            wrong = copy.deepcopy(decision)
            wrong["decisions"][0][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                runner.validate_decisions(phase, declaration, wrong, "protocol")
        for plans in ([], [plan, plan]):
            with self.assertRaises(ValueError):
                runner.validate_decisions(phase, declaration, {**decision, "decisions": plans}, "protocol")
        for status in ("learning", "learning_closed"):
            with self.assertRaises(ValueError):
                runner.validate_decisions({**phase, "status": status}, declaration, decision, "protocol")
        solved = copy.deepcopy(phase)
        solved["cases"]["case"]["status"] = "validation99"
        with self.assertRaises(ValueError):
            runner.validate_decisions(solved, declaration, decision, "protocol")

    def test_compiled_extension_matches_uninterrupted_state_then_censors_honestly(self):
        torch.set_num_threads(1)
        full = swap(tiny(), "execution", ResidualMaxAutotune(device="cpu"))
        initial = swap(full, "budget.updates", 2)
        calls = []
        original_evaluate = ResidualCompiledStepper.evaluate

        def score(self, split):
            calls.append(split)
            if split != "validation":
                raise AssertionError("Final test opened during continuation")
            return original_evaluate(self, split)

        with tempfile.TemporaryDirectory() as temporary, patch.object(ResidualCompiledStepper, "evaluate", score):
            root = Path(temporary)
            _, straight = train(full, root / "straight")
            result, continued = train(initial, root / "runs/case")
            task = build_task(initial.benchmark, initial.seeds.data, torch.device("cpu"))
            declaration = {"python": platform.python_version(), "torch": str(torch.__version__),
                           "cases": [{"label": "case", "fingerprints": task.fingerprints}]}
            (root / "protocol.json").write_text("{}\n")
            digest = runner.sha(continued.path / "checkpoint.pt")
            phase = {"status": "awaiting_continuation", "cases": {"case": {
                "status": "budget", "stop": result["stop"], "elapsed_seconds": 1., "checkpoint_sha256": digest}}}
            runner.write(root / "runs/phase.json", phase)
            decision = {"protocol_sha256": runner.sha(root / "protocol.json"), "decisions": [{
                "label": "case", "step": 2, "checkpoint_sha256": digest, "action": "extend",
                "updates": 6, "reason": "Test an exact compiled continuation across an epoch remainder"}]}
            path = root / "extend.json"
            runner.write(path, decision)
            with patch.object(runner, "frozen", return_value=declaration), \
                 patch.object(runner, "load", return_value=SimpleNamespace(experiments={"case": initial}, source="fixture")):
                runner.continue_cohort(root, path)
                a, b = straight.checkpoint("cpu"), continued.checkpoint("cpu")
                for key in ("model", "optimizer", "permutation"):
                    torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
                for key in ("step", "cursor", "examples_seen", "last_batch_size"):
                    self.assertEqual(a[key], b[key])
                self.assertEqual(straight.records("gradients"), continued.records("gradients"))
                saved = continued.path / "continuation_endpoints/step-00000002.pt"
                self.assertEqual(runner.sha(saved), digest)
                with self.assertRaises(ValueError):
                    runner.continue_cohort(root, path)
                digest = runner.sha(continued.path / "checkpoint.pt")
                decision["decisions"] = [{"label": "case", "step": 6, "checkpoint_sha256": digest,
                                           "action": "censor", "reason": "Test fixture ends unresolved"}]
                path = root / "censor.json"
                runner.write(path, decision)
                runner.continue_cohort(root, path)
            phase = json.loads((root / "runs/phase.json").read_text())
            self.assertEqual(phase["status"], "learning_closed")
            self.assertEqual(phase["cases"]["case"]["status"], "censored")
            self.assertEqual(phase["cases"]["case"]["stop"], {"step": 6, "reason": "budget"})
            self.assertEqual(runner.sha(continued.path / "checkpoint.pt"), digest)
            self.assertEqual(len(continued.records("gradients")), 6)
            self.assertEqual(set(calls), {"validation"})

    def test_final_test_disclosure_blocks_any_further_learning(self):
        declaration = {"python": platform.python_version(), "torch": str(torch.__version__)}
        with tempfile.TemporaryDirectory() as temporary, patch.object(runner, "frozen", return_value=declaration):
            (Path(temporary) / "results.json").write_text("{}\n")
            with self.assertRaisesRegex(ValueError, "final test"):
                runner.continue_cohort(Path(temporary), Path(temporary) / "not-read.json")


if __name__ == "__main__":
    unittest.main()
