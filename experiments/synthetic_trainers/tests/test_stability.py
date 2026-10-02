"""Frozen campaigns reject changed protocols and preserve complete negative runs."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from experiments.synthetic_trainers.paper_reproduction.diagnostics import DiagnosticsConfig
from experiments.synthetic_trainers.paper_reproduction.grokking import RunConfig
from experiments.synthetic_trainers.persistence import PersistenceConfig
from experiments.synthetic_trainers.stability import freeze, run_stage, validate_complete
from experiments.synthetic_trainers.stability_protocol import calibration_recipes


class StabilityTests(unittest.TestCase):
    def base(self):
        return RunConfig(model="gptmini", optimizer="amsgradw", prime=7, train_fraction=.5,
                         width=8, heads=1, layers=1, steps=6, eval_every=2, batch_size=8, device="cpu")

    def test_four_real_controls_keep_negative_results_and_resume_completed_runs(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            recipes = calibration_recipes(self.base())
            manifest = freeze(root, recipes, DiagnosticsConfig(2, True), PersistenceConfig())
            state = run_stage(root, manifest)
            self.assertEqual(state["completed_runs"], 4)
            self.assertEqual(state["status"], "complete")
            self.assertTrue(all(not row["assessment"]["persistent_final_performance"] for row in state["runs"]))
            snapshots = [(root / recipe["name"] / "checkpoint.pt").read_bytes() for recipe in recipes]
            frozen = freeze(root, recipes, DiagnosticsConfig(2, True), PersistenceConfig(), resume=True)
            self.assertEqual(run_stage(root, frozen), state)
            self.assertEqual(snapshots, [(root / recipe["name"] / "checkpoint.pt").read_bytes() for recipe in recipes])
            exposures = [row["final"]["epochs_seen"] for row in state["runs"]]
            self.assertAlmostEqual(exposures[0], 2)
            self.assertAlmostEqual(exposures[1], 48 / 21)
            report = json.loads((root / recipes[0]["name"] / "measurements.json").read_text())
            for mutation in ("history", "examples", "sources"):
                with self.subTest(mutation=mutation):
                    corrupt = deepcopy(report)
                    if mutation == "history":
                        del corrupt["history"][1]
                    elif mutation == "examples":
                        corrupt["history"][1]["heldout"]["examples"] -= 1
                    else:
                        corrupt["plan"]["source_hashes"]["different"] = "bad"
                    with self.assertRaises(ValueError):
                        validate_complete(corrupt, recipes[0], manifest)

    def test_configuration_criterion_sources_and_environment_cannot_change(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            recipes = calibration_recipes(self.base())
            original = freeze(root, recipes, DiagnosticsConfig(2), PersistenceConfig())
            with self.assertRaises(ValueError):
                freeze(root, calibration_recipes(replace(self.base(), steps=7)),
                       DiagnosticsConfig(2), PersistenceConfig(), resume=True)
            with self.assertRaises(ValueError):
                freeze(root, recipes, DiagnosticsConfig(2), PersistenceConfig(tail_steps=1), resume=True)
            for field in ("source_hashes", "environment"):
                damaged = deepcopy(original)
                damaged[field]["different"] = "bad"
                (root / "plan.json").write_text(json.dumps(damaged))
                with self.assertRaises(ValueError):
                    freeze(root, recipes, DiagnosticsConfig(2), PersistenceConfig(), resume=True)
