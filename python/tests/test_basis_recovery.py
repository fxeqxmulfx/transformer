"""Interrupted Basis learning resumes its exact state and keeps the raw evidence."""

import copy
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
from lab.infrastructure.engine.basis_residual import BasisTraining
from lab.infrastructure.engine.residual import ResidualCompiledStepper
from lab.infrastructure.store import RunDirectory
from test_basis_continuation import HERE, module, runner
from test_residual_basis import tiny
from test_residual_transformer import train

with patch.dict("sys.modules", {"continuation": runner}):
    recovery = module("basis_recovery", HERE / "analysis/recovery.py")


class BasisRecoveryTests(unittest.TestCase):
    def test_interrupted_fullgraph_run_matches_uninterrupted_and_preserves_trailing_logs(self):
        torch.set_num_threads(1)
        spec = swap(tiny(), "execution", ResidualMaxAutotune(device="cpu"))
        original_save, original_evaluate = BasisTraining.save, ResidualCompiledStepper.evaluate

        def interrupt(trainer, step):
            original_save(trainer, step)
            if step == 4:
                raise KeyboardInterrupt("Simulated interruption after the saved fourth update")

        def validation_only(stepper, split):
            self.assertEqual(split, "validation")
            return original_evaluate(stepper, split)

        with tempfile.TemporaryDirectory() as temporary, \
             patch.object(ResidualCompiledStepper, "evaluate", validation_only):
            root = Path(temporary)
            _, straight = train(spec, root / "straight")
            with patch.object(BasisTraining, "save", interrupt), self.assertRaises(KeyboardInterrupt):
                train(spec, root / "runs/case")
            directory = RunDirectory(root / "runs/case")
            # Records from an unsaved update are evidence, but cannot advance the restored state.
            directory.record("gradients", {**directory.records("gradients")[-1], "step": 5})
            directory.record("history", {**directory.records("history")[-1], "step": 5})
            digest = recovery.sha(directory.path / "checkpoint.pt")
            phase = {"status": "learning", "cases": {"case": {"status": "learning"}},
                     "continuations": [{"decision_sha256": "previous", "decision": {"decisions": [
                         {"label": "case", "action": "extend", "step": 2, "updates": 6}]}}]}
            recovery.write(root / "runs/phase.json", phase)
            recovery.write(root / "protocol.json", {})
            decision = {"label": "case", "step": 4, "updates": 6, "checkpoint_sha256": digest,
                        "protocol_sha256": recovery.sha(root / "protocol.json"),
                        "phase_sha256": recovery.sha(root / "runs/phase.json"),
                        "continuation_sha256": "previous", "reason": "Recover the interrupted fixture"}
            path = root / "recovery.json"
            recovery.write(path, decision)
            task = build_task(spec.benchmark, spec.seeds.data, torch.device("cpu"))
            declaration = {"python": platform.python_version(), "torch": str(torch.__version__),
                           "cases": [{"label": "case", "data": {"fingerprints": task.fingerprints}}]}
            study = SimpleNamespace(experiments={"case": spec}, source="test fixture")
            recovery.validate_recovery(phase, decision, directory, study)
            for key, value in (("updates", 8), ("step", 3), ("checkpoint_sha256", "wrong"),
                               ("continuation_sha256", "wrong"), ("reason", "")):
                with self.subTest(key=key), self.assertRaises(ValueError):
                    recovery.validate_recovery(phase, {**decision, key: value}, directory, study)
            for status in ("validation99", "censored", "budget"):
                closed = copy.deepcopy(phase)
                closed["cases"]["case"]["status"] = status
                with self.assertRaises(ValueError):
                    recovery.validate_recovery(closed, decision, directory, study)
            with patch.object(recovery, "HERE", root), patch.object(recovery, "frozen", return_value=declaration), \
                 patch.object(recovery, "load", return_value=study):
                recovery.recover(path)
                with self.assertRaisesRegex(ValueError, "phase changed"):
                    recovery.recover(path)
            actual, expected = directory.checkpoint("cpu"), straight.checkpoint("cpu")
            clock_keys = {"training_seconds", "diagnostic_seconds", "wall_seconds"}
            torch.testing.assert_close({k: v for k, v in actual.items() if k not in clock_keys},
                                       {k: v for k, v in expected.items() if k not in clock_keys},
                                       rtol=0, atol=0)
            self.assertEqual(directory.records("gradients"), straight.records("gradients"))
            self.assertEqual([r["step"] for r in directory.records("history")], [0, 2, 4, 6])
            for a, b in zip(directory.records("history"), straight.records("history"), strict=True):
                self.assertEqual(a["validation"], b["validation"])
            archive = root / "runs/recoveries" / recovery.sha(path)
            self.assertEqual(recovery.sha(archive / "checkpoint.pt"), digest)
            self.assertEqual(len((archive / "gradients.jsonl").read_text().splitlines()), 5)
            final = json.loads((root / "runs/phase.json").read_text())
            self.assertEqual(final["status"], "awaiting_continuation")
            self.assertEqual(final["cases"]["case"]["stop"], {"step": 6, "reason": "budget"})
            self.assertEqual(len(final["recoveries"]), 1)

    def test_recovery_cannot_reopen_final_test_or_changed_phase(self):
        declaration = {"python": platform.python_version(), "torch": str(torch.__version__)}
        with tempfile.TemporaryDirectory() as temporary, patch.object(recovery, "HERE", Path(temporary)), \
             patch.object(recovery, "frozen", return_value=declaration):
            root = Path(temporary)
            (root / "results.json").write_text("{}\n")
            with self.assertRaisesRegex(ValueError, "final test"):
                recovery.recover(root / "not-read.json")
            (root / "results.json").unlink()
            recovery.write(root / "protocol.json", {})
            recovery.write(root / "runs/phase.json", {})
            recovery.write(root / "decision.json", {
                "protocol_sha256": recovery.sha(root / "protocol.json"), "phase_sha256": "stale"})
            with self.assertRaisesRegex(ValueError, "phase changed"):
                recovery.recover(root / "decision.json")


if __name__ == "__main__":
    unittest.main()
