"""Confirmation campaigns preserve paired data and verified completed work."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from experiments.synthetic_trainers.paper_reproduction.grokking import RunConfig, train
from experiments.synthetic_trainers.reproduction import run_campaign


class ReproductionTests(unittest.TestCase):
    def test_adoption_resume_and_integrity_for_three_real_model_optimizer_controls(self):
        base = RunConfig(prime=7, train_fraction=.5, width=8, layers=1, heads=1,
                         batch_size=4, steps=2, eval_every=1, device="cpu")
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            train(base, directory / "reference-adamw-seed0")
            with self.assertRaises(FileExistsError):
                run_campaign(directory, base, (0,))
            runs = run_campaign(directory, base, (0,), adopt_completed=True)
            manifest = json.loads((directory / "plan.json").read_text())
            self.assertEqual(manifest["adopted_completed_runs"], ["reference-adamw-seed0"])
            self.assertEqual(len(runs), 3)
            reports = [json.loads(Path(row["result"]).read_text()) for row in runs]
            self.assertEqual(len({row["plan"]["corpus"]["train_fingerprint"] for row in reports}), 1)
            self.assertEqual(len({row["plan"]["corpus"]["heldout_fingerprint"] for row in reports}), 1)
            self.assertEqual({(row["plan"]["config"]["model"], row["plan"]["config"]["optimizer"]) for row in reports},
                             {("reference", "adamw"), ("gptmini", "adamw"), ("gptmini", "amsgradw")})
            snapshots = {row["name"]: (directory / row["name"] / "checkpoint.pt").read_bytes() for row in runs}
            run_campaign(directory, base, (0,), resume=True)
            self.assertEqual(snapshots, {name: (directory / name / "checkpoint.pt").read_bytes() for name in snapshots})
            with self.assertRaises(ValueError):
                run_campaign(directory, replace(base, steps=3), (0,), resume=True)
            manifest["source_hashes"]["corrupted"] = "wrong"
            (directory / "plan.json").write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                run_campaign(directory, base, (0,), resume=True)
