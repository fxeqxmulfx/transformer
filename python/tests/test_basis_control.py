"""The original Basis model keeps its data, objective and complete-state stopping."""

import copy
import json
from pathlib import Path
import tempfile
import importlib.util
import sys
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from examples import gptmini
from lab.dsl import BasisControl, Experiment, ResidualMaxAutotune, swap
from lab.infrastructure.benchmarks import build_task
from lab.infrastructure.benchmarks.synthetic.vocabulary import IGNORE
from lab.infrastructure.engine.residual import ResidualCompiledStepper, ResidualEagerStepper
from lab.infrastructure.loader import load
from lab.infrastructure.nn import build_model
from test_residual_basis import tiny
from test_residual_transformer import train

ROOT = Path(__file__).resolve().parents[2]


def control():
    original = tiny()
    return Experiment(**{**vars(original), "benchmark": BasisControl(**vars(original.benchmark)),
                         "model": swap(gptmini(8, 1, 2), "context", original.benchmark.context)})


class BasisControlTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_historical_architecture_and_frozen_current_data_are_unchanged(self):
        study = load(ROOT / "experiments/basis_control")
        original = load(ROOT / "experiments/basis")
        frozen = json.loads((ROOT / "experiments/basis_residual/protocol.json").read_text())
        for seed in (0, 1):
            run = study.experiments[f"gptmini-hard-recall-s{seed}"]
            old = original.experiments[f"hard-large-recall-seed{seed}"]
            self.assertEqual(run.model, old.model)
            self.assertEqual(vars(run.benchmark), vars(old.benchmark))
            self.assertEqual(run.seeds, old.seeds)
            self.assertEqual(run.schedule, old.schedule)
            self.assertEqual(run.budget.batch, old.budget.batch)
            self.assertEqual(run.optimizer.lr, old.optimizer.lr)
            task = build_task(run.benchmark, run.seeds.data, torch.device("cpu"))
            reference = next(c for c in frozen["cases"] if c["label"] == f"canonical-large-recall-s{seed}")
            self.assertEqual(task.fingerprints, reference["data"]["fingerprints"])
        with self.assertRaisesRegex(ValueError, "declarative Transformer"):
            swap(tiny(), "benchmark", BasisControl(**vars(tiny().benchmark))).check()
        with self.assertRaisesRegex(ValueError, "Schedule-Free AdamW"):
            swap(run, "optimizer", old.optimizer).check()

    def test_original_optimized_readout_matches_full_masked_loss_and_gradients(self):
        spec = control()
        task = build_task(spec.benchmark, spec.seeds.data, torch.device("cpu"))
        model = build_model(spec.model, task.vocab, spec.seeds.model).double()
        reference = copy.deepcopy(model)
        batch = task.inputs(torch.arange(4), static=True)
        output, targets = task.forward(model, batch, supervised=True)
        actual = task.loss(output, targets)
        logits = reference(batch[:, 0])
        expected = F.cross_entropy(logits.flatten(0, 1), batch[:, 1].flatten(), ignore_index=IGNORE)
        torch.testing.assert_close(actual, expected, rtol=1e-13, atol=1e-13)
        actual.backward()
        expected.backward()
        for a, b in zip(model.parameters(), reference.parameters(), strict=True):
            torch.testing.assert_close(a.grad, b.grad, rtol=1e-11, atol=1e-13)

    def test_compiled_control_resumes_full_state_without_reading_final_test(self):
        spec = swap(control(), "execution", ResidualMaxAutotune(device="cpu"))
        evaluate = ResidualCompiledStepper.evaluate

        def validation_only(stepper, name):
            self.assertEqual(name, "validation")
            self.assertIsNone(stepper.conditioner)
            return evaluate(stepper, name)

        with tempfile.TemporaryDirectory() as directory, \
             patch.object(ResidualCompiledStepper, "evaluate", validation_only):
            root = Path(directory)
            _, straight = train(spec, root / "straight")
            train(swap(spec, "budget.updates", 2), root / "continued")
            result, continued = train(spec, root / "continued")
            a, b = straight.checkpoint("cpu"), continued.checkpoint("cpu")
            clocks = {"training_seconds", "diagnostic_seconds", "wall_seconds"}
            torch.testing.assert_close({k: v for k, v in a.items() if k not in clocks},
                                       {k: v for k, v in b.items() if k not in clocks}, rtol=0, atol=0)
            self.assertEqual(straight.records("gradients"), continued.records("gradients"))
            self.assertEqual(result["stop"], {"step": 6, "reason": "budget"})

    def test_first99_control_stops_at_current_state_and_does_not_reopen_it(self):
        calls = []

        def score(stepper, name):
            self.assertEqual(name, "validation")
            calls.append(name)
            return {"sequence_accuracy": .98 if len(calls) == 1 else .99, "loss": .1}

        with tempfile.TemporaryDirectory() as directory, patch.object(ResidualEagerStepper, "evaluate", score):
            result, run = train(control(), Path(directory))
            self.assertEqual(result["stop"], {"step": 2, "reason": "validation99"})
            saved = copy.deepcopy(run.checkpoint("cpu"))
            repeated, run = train(control(), Path(directory))
            self.assertEqual(repeated["stop"], result["stop"])
            self.assertEqual(calls, ["validation", "validation"])
            torch.testing.assert_close(run.checkpoint("cpu"), saved, rtol=0, atol=0)
            self.assertNotIn("best", result)

    def test_control_report_stays_sealed_before_complete_closure(self):
        folder = ROOT / "experiments/basis_control"

        def module(name, path):
            spec = importlib.util.spec_from_file_location(name, path)
            value = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(value)
            return value

        protocol = module("basis_control_protocol", folder / "protocol.py")
        with patch.dict(sys.modules, {"protocol": protocol}):
            report = module("basis_control_report", folder / "report.py")
            cohort = module("basis_control_cohort", folder / "cohort.py")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "runs").mkdir()
            declaration = {"cases": [{"label": "case"}]}
            for status in ("learning", "interrupted", "awaiting_continuation"):
                protocol.write(root / "runs/phase.json", {"status": status, "cases": {}})
                with patch.object(report, "HERE", root), patch.object(report, "frozen", return_value=declaration), \
                     self.assertRaisesRegex(ValueError, "sealed"):
                    report.report()
            (root / "results.json").write_text("{}\n")
            with patch.object(cohort, "HERE", root), patch.object(cohort, "frozen", return_value=declaration), \
                 self.assertRaisesRegex(ValueError, "already open"):
                cohort.run()


if __name__ == "__main__":
    unittest.main()
