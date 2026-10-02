"""Portable archives preserve real histories and reject partial diagnostics."""

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
from experiments.synthetic_trainers.stability_analysis import collapse_neighborhoods, diagnostic_metrics
from experiments.synthetic_trainers.stability_integrity import validate_logs
from experiments.synthetic_trainers.stability_protocol import calibration_recipes
from experiments.synthetic_trainers.stability_report import file_hashes, parse_rows, save_run, verify_archive


class StabilityReportTests(unittest.TestCase):
    def real_stage(self, root):
        config = RunConfig(model="gptmini", optimizer="amsgradw", prime=7, train_fraction=.5,
                           width=8, heads=1, layers=1, steps=6, eval_every=2, batch_size=8, device="cpu")
        recipes = calibration_recipes(config)
        plan = freeze(root, recipes, DiagnosticsConfig(2, True), PersistenceConfig())
        run_stage(root, plan)
        return plan

    def test_archive_survives_deleting_training_and_verifies_without_torch(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, archive = Path(temporary) / "source", Path(temporary) / "archive"
            self.real_stage(source)
            original = (source / "short-lr001/measurements.json").read_bytes()
            summary = save_run(source, "short-lr001", archive, render=False)
            self.assertTrue(summary["complete_run"])
            self.assertFalse(summary["assessment"]["persistent_final_performance"])
            self.assertEqual((archive / "measurements.json").read_bytes(), original)
            self.assertEqual(len((archive / "metrics.csv").read_text().splitlines()), 5)
            shutil.rmtree(source)
            self.assertEqual(verify_archive(archive), summary)
            result = subprocess.run([sys.executable, "-c",
                "import sys; from experiments.synthetic_trainers.stability_report import verify_archive; "
                "verify_archive(sys.argv[1]); assert 'torch' not in sys.modules; print('offline verified')", str(archive)],
                check=True, capture_output=True, text=True)
            self.assertEqual(result.stdout.strip(), "offline verified")
            with self.assertRaises(FileExistsError):
                save_run(source, "short-lr001", archive, render=False)
            csv_path = archive / "metrics.csv"
            original_csv = csv_path.read_text()
            lines = original_csv.splitlines()
            lines[-1] = lines[-2]
            csv_path.write_text("\n".join(lines) + "\n")
            hash_path = archive / "artifact-hashes.json"
            hashes = json.loads(hash_path.read_text())
            hashes["files"] = file_hashes(archive)
            hash_path.write_text(json.dumps(hashes))
            with self.assertRaisesRegex(ValueError, "metrics.csv"):
                verify_archive(archive)
            csv_path.write_text(original_csv)
            hashes["files"] = file_hashes(archive)
            hash_path.write_text(json.dumps(hashes))
            with (archive / "probes.jsonl").open("a") as stream:
                stream.write('{}\n')
            with self.assertRaisesRegex(ValueError, "hashes"):
                verify_archive(archive)

    def test_missing_and_corrupt_diagnostics_cannot_be_archived(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source"
            manifest = self.real_stage(source)
            root = source / "short-lr001"
            report = json.loads((root / "measurements.json").read_text())
            diagnostics = parse_rows((root / "diagnostics.jsonl").read_bytes())
            probes = parse_rows((root / "probes.jsonl").read_bytes())
            for case in ("missing", "batch", "moment", "temperature", "probe"):
                with self.subTest(case=case):
                    bad, neighbors = deepcopy(diagnostics), deepcopy(probes)
                    if case == "missing":
                        bad.pop()
                    elif case == "batch":
                        bad[0]["batch_size"] += 1
                    elif case == "moment":
                        del next(iter(bad[0]["parameters"].values()))["maximum_l2"]
                    elif case == "temperature":
                        next(iter(bad[0]["temperatures"].values()))["inverse_temperature"][0] *= 2
                    else:
                        neighbors.pop()
                    with self.assertRaises(ValueError):
                        validate_logs(report, bad, neighbors, manifest)
            (root / "diagnostics.jsonl").write_text("\n".join(json.dumps(p) for p in diagnostics[:-1]) + "\n")
            with self.assertRaisesRegex(ValueError, "Diagnostic observations"):
                save_run(source, "short-lr001", Path(temporary) / "incomplete", render=False)
            self.assertFalse((Path(temporary) / "incomplete").exists())

    def test_neighbor_support_separates_transient_and_multistep_failures(self):
        def point(step, accuracy, *, answer=None, eos=1):
            return {"step": step, "train": {"accuracy": 1}, "last_batch_size": 48 if step % 250 == 0 else 512,
                    "heldout": {"accuracy": accuracy, "answer_accuracy": accuracy if answer is None else answer,
                                "EOS_accuracy": eos}}
        report = {"plan": {"config": {"target": .99, "patience": 2}},
            "history": [point(0, 0), point(250, 1), point(500, 1), point(750, .4),
                        point(1000, .2), point(1250, .4), point(1500, .2, answer=1, eos=.2)]}
        probes = [point(749, 1), point(751, 1), point(999, .3), point(1001, .3), point(1249, .1), point(1251, 1)]
        result = collapse_neighborhoods(report, probes, [])
        self.assertEqual(result["failures_after_confirmation"], 4)
        self.assertEqual(result["isolated_at_canonical_observation"], 1)
        self.assertEqual(result["still_below_target_both_neighbors"], 1)
        self.assertEqual(result["recovered_by_next_update"], 2)
        self.assertEqual(result["numeric_answer_failures"], 3)
        self.assertEqual(result["EOS_failures"], 1)
        self.assertEqual(result["missing_neighbor_support"], 1)

    def test_joint_norms_use_the_euclidean_parameter_vector(self):
        norms = lambda size: {key: size for key in ("parameter_l2", "parameter_before_l2", "gradient_l2",
                               "update_l2", "m_l2", "v_l2", "maximum_l2", "maximum_min", "maximum_max")}
        row = {"step": 1, "batch_size": 48, "epoch_tail": True, "epoch_wraps_in_batch": 0,
               "learning_rate": .001, "answer_loss": 1, "EOS_loss": .1,
               "parameters": {"first": norms(3), "second": norms(4)}, "temperatures": {}}
        measured = diagnostic_metrics(row)
        self.assertEqual(measured["gradient_l2"], 5)
        self.assertEqual(measured["joint_relative_update"], 1)
        self.assertIsNone(measured["temperature_max"])
