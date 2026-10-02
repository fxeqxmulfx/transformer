"""AdamW controls retain optimizer provenance, frozen gates and failed repeats."""

from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import torch

from experiments.synthetic_trainers.confirmation_layout import validate_plan
from experiments.synthetic_trainers.confirmation_protocol import freeze as freeze_confirmation
from experiments.synthetic_trainers.confirmation_report import assemble as confirm, verify_confirmation
from experiments.synthetic_trainers.paper_reproduction.diagnostics import DiagnosticsConfig
from experiments.synthetic_trainers.paper_reproduction.grokking import RunConfig
from experiments.synthetic_trainers.persistence import PersistenceConfig
from experiments.synthetic_trainers.stability import freeze, run_stage
from experiments.synthetic_trainers.stability_comparison import assemble, verify_comparison
from experiments.synthetic_trainers.stability_confirmation import run_confirmation
from experiments.synthetic_trainers.stability_integrity import validate_logs
from experiments.synthetic_trainers.stability_report import save_run
from experiments.synthetic_trainers.tests.test_confirmation import calibration_fixture


class AdamWStabilityTests(unittest.TestCase):
    def test_paired_optimizer_archive_identifies_both_controls_and_retains_failures(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = RunConfig(model="gptmini", optimizer="adamw", prime=7, train_fraction=.5,
                width=8, heads=1, layers=1, steps=6, eval_every=2, batch_size=8, device="cpu")
            recipes = [{"name": optimizer, "config": asdict(replace(base, optimizer=optimizer)),
                        "changed_mechanism": "optimizer"} for optimizer in ("adamw", "amsgradw")]
            manifest = freeze(root / "source", recipes, DiagnosticsConfig(2, True, True), PersistenceConfig())
            run_stage(root / "source", manifest)
            adamw = root / "source/adamw"
            report = json.loads((adamw / "measurements.json").read_text())
            self.assertEqual(report["plan"]["optimizer"]["betas"], [.9, .98])
            self.assertTrue(report["plan"]["optimizer"]["bias_correction"])
            diagnostics = [json.loads(line) for line in (adamw / "diagnostics.jsonl").read_text().splitlines()]
            checkpoint = torch.load(adamw / "checkpoint.pt", weights_only=True)
            ids = checkpoint["optimizer"]["param_groups"][0]["params"]
            for norms, index in zip(diagnostics[-1]["parameters"].values(), ids):
                state = checkpoint["optimizer"]["state"][index]
                self.assertAlmostEqual(norms["exp_avg_l2"], float(state["exp_avg"].norm()))
                self.assertAlmostEqual(norms["exp_avg_sq_l2"], float(state["exp_avg_sq"].norm()))
                self.assertNotIn("maximum_l2", norms)
            bad = deepcopy(diagnostics)
            del next(iter(bad[0]["parameters"].values()))["exp_avg_sq_l2"]
            probes = [json.loads(line) for line in (adamw / "probes.jsonl").read_text().splitlines()]
            gradients = [json.loads(line) for line in (adamw / "gradients.jsonl").read_text().splitlines()]
            with self.assertRaisesRegex(ValueError, "moment"):
                validate_logs(report, bad, probes, manifest, gradients)
            paths = []
            for recipe in recipes:
                path = root / "individual" / recipe["name"]
                save_run(root / "source", recipe["name"], path, render=False)
                paths.append(path)
            summary = assemble(paths, root / "portable", render=False)
            self.assertEqual(summary["optimizers"], ["adamw", "amsgradw"])
            self.assertEqual(summary["optimizer_label"], "AdamW / raw AMSGradW")
            self.assertEqual([row["optimizer"] for row in summary["rows"]], summary["optimizers"])
            self.assertFalse(summary["ready_to_freeze_independent_confirmation"])
            self.assertEqual(summary["total_updates"], 12)
            text = (root / "portable/REPORT.md").read_text()
            self.assertIn("# Complete AdamW / raw AMSGradW stability calibration", text)
            self.assertIn("optimizer", (root / "portable/comparison.csv").read_text().splitlines()[0])
            shutil.rmtree(root / "source")
            shutil.rmtree(root / "individual")
            self.assertEqual(verify_comparison(root / "portable"), summary)
            subprocess.run([sys.executable, "-c", "import sys; "
                "from experiments.synthetic_trainers.stability_comparison import verify_comparison; "
                "verify_comparison(sys.argv[1]); assert 'torch' not in sys.modules", str(root / "portable")],
                check=True, capture_output=True, text=True)

    def test_negative_adamw_calibration_still_cannot_launch_confirmation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            calibration = calibration_fixture(root, optimizer="adamw")
            with self.assertRaisesRegex(ValueError, "passing recipe"):
                freeze_confirmation(root / "confirmation", calibration, "short-lr001", render=False)
            self.assertFalse((root / "confirmation").exists())

    def test_adamw_confirmation_freezes_optimizer_and_keeps_all_six_dense_negative_runs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            # Hypothetical passing calibration tests the gate, not scientific learning.
            calibration = calibration_fixture(root, hypothetical_passing=True,
                                              optimizer="adamw", trace_gradients=True)
            stage = root / "confirmation"
            plan = freeze_confirmation(stage, calibration, "short-lr001", render=False)
            self.assertEqual(plan["planned_runs"], 6)
            self.assertTrue(all(row["config"]["optimizer"] == "adamw" for row in plan["recipes"]))
            changed = deepcopy(plan)
            changed["recipes"][0]["config"]["optimizer"] = "amsgradw"
            with self.assertRaisesRegex(ValueError, "changed more"):
                validate_plan(changed, calibration)
            result = run_confirmation(stage, plan)
            self.assertFalse(result["repeatable_stable_benchmark"])
            summary = confirm(stage, root / "portable", render=False)
            self.assertEqual(summary["optimizers"], ["adamw"])
            self.assertEqual(summary["optimizer_label"], "AdamW")
            self.assertEqual(summary["completed_runs"], 6)
            self.assertEqual(summary["total_updates"], 48)
            self.assertFalse(summary["architecture_comparison_ready"])
            for row in plan["recipes"]:
                trace = root / "portable/runs" / row["name"] / "gradients.jsonl"
                self.assertEqual([json.loads(line)["step"] for line in trace.read_text().splitlines()],
                                 list(range(1, 9)))
            self.assertIn("# Independent AdamW / GPTMini confirmation",
                          (root / "portable/REPORT.md").read_text())
            shutil.rmtree(stage)
            shutil.rmtree(calibration)
            self.assertEqual(verify_confirmation(root / "portable"), summary)
            subprocess.run([sys.executable, "-c", "import sys; "
                "from experiments.synthetic_trainers.confirmation_report import verify_confirmation; "
                "verify_confirmation(sys.argv[1]); assert 'torch' not in sys.modules", str(root / "portable")],
                check=True, capture_output=True, text=True)
