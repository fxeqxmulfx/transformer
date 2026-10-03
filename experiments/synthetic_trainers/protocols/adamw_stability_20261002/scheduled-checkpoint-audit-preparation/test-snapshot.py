"""Actual native state checks for completed fixed-schedule CPU cases."""

import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import torch

from experiments.synthetic_trainers.paper_reproduction.diagnostics import DiagnosticsConfig
from experiments.synthetic_trainers.persistence import PersistenceConfig
from experiments.synthetic_trainers.protocols.adamw_stability_20261002.audit_scheduled_checkpoint import audit
from experiments.synthetic_trainers.scheduled_protocol import freeze_pair, run_pair
from experiments.synthetic_trainers.scheduled_rates import expected_rate
from experiments.synthetic_trainers.scheduled_training import ScheduledRunConfig


class ScheduledCheckpointAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.stage = cls.root / "actual-cpu-pair"
        config = ScheduledRunConfig(model="gptmini", optimizer="adamw", prime=7,
            train_fraction=.5, width=8, layers=1, heads=1, steps=12, batch_size=4,
            eval_every=4, learning_rate=.0003, weight_decay=.1, warmup_steps=10,
            device="cpu", anneal_start=10, anneal_end=11, final_rate_factor=.1)
        cls.plan = freeze_pair(cls.stage, config, DiagnosticsConfig(4, True, True),
            PersistenceConfig(plateau_steps=4, plateau_observations=2,
                              confirmation_observations=2, tail_steps=4))
        cls.state = run_pair(cls.stage, cls.plan)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_both_actual_native_checkpoints_use_their_final_rate(self):
        self.assertEqual(self.state["completed_runs"], 2)
        for recipe in self.plan["recipes"]:
            with self.subTest(recipe=recipe["name"]):
                result = audit(self.stage, recipe["name"])
                self.assertFalse(result["scientific_run"])
                self.assertEqual(result["completed_updates"], 12)
                self.assertEqual(result["learning_rate"], expected_rate(recipe["config"], 11))
                self.assertTrue(result["all_model_and_native_tensors_finite"])
                self.assertEqual(result["optimizer_updates_performed"], 0)
                self.assertFalse(result["GPU_context_initialized"])
        self.assertNotEqual(audit(self.stage, "adamw-constant")["learning_rate"],
                            audit(self.stage, "adamw-cosine-tail")["learning_rate"])

    def test_native_state_forgery_is_rejected(self):
        original = torch.load(self.stage / "adamw-cosine-tail/checkpoint.pt",
                              map_location="cpu", weights_only=True)
        for field, message in (("rate", "learning rate"), ("step", "optimizer steps"),
                               ("moment", "nonfinite"), ("model", "nonfinite")):
            with self.subTest(field=field):
                target = self.root / f"forged-{field}"
                shutil.copytree(self.stage, target)
                checkpoint = copy.deepcopy(original)
                if field == "rate":
                    checkpoint["optimizer"]["param_groups"][0]["lr"] = .0003
                elif field == "model":
                    next(iter(checkpoint["model"].values())).reshape(-1)[0] = float("nan")
                else:
                    state = next(iter(checkpoint["optimizer"]["state"].values()))
                    if field == "step":
                        state["step"].fill_(11)
                    else:
                        state["exp_avg"].reshape(-1)[0] = float("nan")
                torch.save(checkpoint, target / "adamw-cosine-tail/checkpoint.pt")
                with self.assertRaisesRegex(ValueError, message):
                    audit(target, "adamw-cosine-tail")

    def test_incomplete_case_is_rejected(self):
        target = self.root / "partial-case"
        shutil.copytree(self.stage, target)
        (target / "adamw-constant/measurements.json").unlink()
        with self.assertRaisesRegex(ValueError, "complete frozen case"):
            audit(target, "adamw-constant")

    def test_cli_preserves_an_existing_receipt(self):
        destination = self.root / "native-audit.json"
        command = [sys.executable, "-m", "experiments.synthetic_trainers.protocols.adamw_stability_20261002.audit_scheduled_checkpoint",
                   "--stage", str(self.stage), "--case", "adamw-cosine-tail", "--output", str(destination)]
        subprocess.run(command, check=True, capture_output=True, text=True)
        first = destination.read_bytes()
        self.assertEqual(json.loads(first)["completed_updates"], 12)
        repeated = subprocess.run(command, capture_output=True, text=True)
        self.assertNotEqual(repeated.returncode, 0)
        self.assertIn("fresh output", repeated.stderr)
        self.assertEqual(destination.read_bytes(), first)


if __name__ == "__main__":
    unittest.main()
