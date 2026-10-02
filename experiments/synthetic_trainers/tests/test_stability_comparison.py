"""Complete comparisons cannot drop failures, mix frozen plans, or imply confirmation."""

from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from experiments.synthetic_trainers.paper_reproduction.diagnostics import DiagnosticsConfig
from experiments.synthetic_trainers.paper_reproduction.grokking import RunConfig
from experiments.synthetic_trainers.persistence import PersistenceConfig
from experiments.synthetic_trainers.stability import freeze, run_stage
from experiments.synthetic_trainers.stability_comparison import (
    assemble, comparison_summary, hashes, verify_comparison,
)
from experiments.synthetic_trainers.stability_protocol import calibration_recipes
from experiments.synthetic_trainers.stability_report import file_hashes, save_run


class StabilityComparisonTests(unittest.TestCase):
    def archive_real_controls(self, root):
        config = RunConfig(model="gptmini", optimizer="amsgradw", prime=7, train_fraction=.5,
                           width=8, heads=1, layers=1, steps=6, eval_every=2, batch_size=8, device="cpu")
        recipes = calibration_recipes(config)
        manifest = freeze(root / "source", recipes, DiagnosticsConfig(2, True), PersistenceConfig())
        run_stage(root / "source", manifest)
        summaries, paths = {}, []
        for recipe in recipes:
            name = recipe["name"]
            path = root / "individual" / name
            summaries[name] = save_run(root / "source", name, path, render=False)
            paths.append(path)
        return manifest, summaries, paths

    def test_portable_complete_comparison_preserves_all_four_negative_runs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, summaries, paths = self.archive_real_controls(root)
            result = assemble(paths, root / "combined", render=False)
            self.assertEqual(result["completed_runs"], 4)
            self.assertEqual(result["total_updates"], 24)
            self.assertEqual(result["total_canonical_observations"], 16)
            self.assertFalse(result["ready_to_freeze_independent_confirmation"])
            self.assertFalse(result["independent_confirmation_complete"])
            self.assertTrue(all(row["long_confirmation_support"] == 0 for row in result["rows"]))
            self.assertTrue(all(row["long_confirmation_training_seconds"] is None for row in result["rows"]))
            for name, summary in summaries.items():
                self.assertEqual(json.loads((root / "combined/runs" / name / "summary.json").read_text()), summary)
            shutil.rmtree(root / "source")
            shutil.rmtree(root / "individual")
            self.assertEqual(verify_comparison(root / "combined"), result)
            verified = subprocess.run([sys.executable, "-c",
                "import sys; from experiments.synthetic_trainers.stability_comparison import verify_comparison; "
                "verify_comparison(sys.argv[1]); assert 'torch' not in sys.modules", str(root / "combined")],
                check=True, capture_output=True, text=True)
            self.assertEqual(verified.returncode, 0)
            csv_path = root / "combined/comparison.csv"
            lines = csv_path.read_text().splitlines()
            csv_path.write_text("\n".join(lines[:-1]) + "\n")
            (root / "combined/artifact-hashes.json").write_text(json.dumps({"files": hashes(root / "combined")}))
            with self.assertRaisesRegex(ValueError, "comparison.csv"):
                verify_comparison(root / "combined")

    def test_missing_duplicate_or_mismatched_archives_cannot_form_a_complete_stage(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, _, paths = self.archive_real_controls(root)
            with self.assertRaisesRegex(ValueError, "every frozen complete run"):
                assemble(paths[:-1], root / "missing", render=False)
            self.assertFalse((root / "missing").exists())
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                assemble([*paths, paths[0]], root / "duplicate", render=False)
            path = paths[-1] / "plan.json"
            plan = json.loads(path.read_text())
            plan["frozen_utc"] = "different_campaign"
            path.write_text(json.dumps(plan))
            hash_path = paths[-1] / "artifact-hashes.json"
            content = json.loads(hash_path.read_text())
            content["files"] = file_hashes(paths[-1])
            hash_path.write_text(json.dumps(content))
            with self.assertRaisesRegex(ValueError, "identical frozen plan"):
                assemble(paths, root / "different", render=False)

    def test_an_early_long_target_streak_does_not_pass_persistence_or_phase_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest, summaries, _ = self.archive_real_controls(Path(temporary))
            # A hypothetical early target event is retained even when its later tail fails.
            modified = deepcopy(summaries)
            modified["short-lr001"]["assessment"]["long_confirmation"] = {
                "onset": 2, "confirmed": 4, "training_seconds": 1.25, "wall_seconds": 2.5}
            result = comparison_summary(manifest, modified)
            row = result["rows"][0]
            self.assertEqual(row["long_confirmation_support"], 1)
            self.assertEqual(row["long_confirmation_training_seconds"], 1.25)
            self.assertFalse(row["persistent_final_performance"])
            self.assertFalse(row["stable_grokking"])
            self.assertFalse(result["ready_to_freeze_independent_confirmation"])
