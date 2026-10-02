"""Independent repeats preserve full budgets, seed exclusions and negative outcomes."""

from copy import deepcopy
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from experiments.synthetic_trainers.confirmation_layout import validate_plan
from experiments.synthetic_trainers.confirmation_protocol import freeze, resume
from experiments.synthetic_trainers.confirmation_report import assemble, derive, verify_confirmation
from experiments.synthetic_trainers.paper_reproduction.diagnostics import DiagnosticsConfig
from experiments.synthetic_trainers.paper_reproduction.grokking import RunConfig
from experiments.synthetic_trainers.persistence import PersistenceConfig
from experiments.synthetic_trainers.stability import freeze as freeze_calibration, run_stage
from experiments.synthetic_trainers.stability_comparison import assemble as compare, hashes
from experiments.synthetic_trainers.stability_confirmation import run_confirmation
from experiments.synthetic_trainers.stability_protocol import calibration_recipes
from experiments.synthetic_trainers.stability_report import save_run, verify_archive


def calibration_fixture(root, *, hypothetical_passing=False, optimizer="amsgradw", trace_gradients=False):
    config = RunConfig(model="gptmini", optimizer="amsgradw", prime=7, train_fraction=.5,
        width=8, heads=1, layers=1, steps=8, eval_every=1, batch_size=8, device="cpu")
    criterion = PersistenceConfig(plateau_steps=1, plateau_observations=2,
                                 confirmation_observations=2, tail_steps=2)
    recipes = calibration_recipes(config)
    for recipe in recipes:
        recipe["config"]["optimizer"] = optimizer
    plan = freeze_calibration(root / "raw", recipes, DiagnosticsConfig(1, True, trace_gradients), criterion)
    run_stage(root / "raw", plan)
    if hypothetical_passing:
        # Hypothetical accuracy fixture tests orchestration, not a learning effect.
        path = root / "raw/short-lr001/measurements.json"
        report = json.loads(path.read_text())
        report["unit_test_fixture"] = "hypothetical_accuracy_for_gate_and_layout_only"
        for point in report["history"]:
            for split, threshold in (("train", 1), ("heldout", 3)):
                accuracy = float(point["step"] >= threshold)
                answer_loss = .01 if accuracy else 1.
                point[split].update(accuracy=accuracy, answer_accuracy=accuracy,
                    EOS_accuracy=1., loss=(answer_loss + .01) / 2, answer_loss=answer_loss, EOS_loss=.01)
        report["final"] = report["history"][-1]
        path.write_text(json.dumps(report))
        (path.parent / "history.jsonl").write_text("".join(json.dumps(p) + "\n" for p in report["history"]))
    paths = []
    for recipe in recipes:
        path = root / "individual" / recipe["name"]
        save_run(root / "raw", recipe["name"], path, render=False)
        paths.append(path)
    compare(paths, root / "calibration", render=False)
    return root / "calibration"


class ConfirmationTests(unittest.TestCase):
    def test_a_complete_negative_calibration_cannot_launch_confirmation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            calibration = calibration_fixture(root)
            with self.assertRaisesRegex(ValueError, "passing recipe"):
                freeze(root / "confirmation", calibration, "short-lr001", render=False)
            self.assertFalse((root / "confirmation").exists())

    def test_seed_exclusions_full_matrix_and_resume_guards(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            calibration = calibration_fixture(root, hypothetical_passing=True)
            for kwargs in ({"model_seeds": (0, 4, 5)}, {"model_seeds": (1, 4, 5)}, {"model_seeds": (4, 4, 5)},
                           {"model_seeds": (4, 5)}, {"data_seeds": (0, 2)},
                           {"data_seeds": (1, 2)}, {"data_seeds": (2, 2)}, {"data_seeds": (2,)}):
                with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                    freeze(root / "invalid", calibration, "short-lr001", render=False, **kwargs)
                self.assertFalse((root / "invalid").exists())
            plan = freeze(root / "confirmation", calibration, "short-lr001", render=False)
            self.assertEqual(plan["planned_runs"], 6)
            self.assertEqual([(r["config"]["seed"], r["config"]["data_seed"]) for r in plan["recipes"]],
                             [(4, 2), (5, 2), (6, 2), (4, 3), (5, 3), (6, 3)])
            self.assertEqual(resume(root / "confirmation"), plan)
            for case in ("missing", "rate", "corpus"):
                bad = deepcopy(plan)
                if case == "missing":
                    bad["recipes"].pop()
                elif case == "rate":
                    bad["recipes"][0]["config"]["learning_rate"] *= 2
                else:
                    bad["recipes"][0]["corpus"]["data_seed"] = 3
                with self.subTest(case=case), self.assertRaises(ValueError):
                    validate_plan(bad, calibration)
            with patch("experiments.synthetic_trainers.confirmation_protocol.confirmation_sources", return_value={}):
                with self.assertRaisesRegex(ValueError, "sources changed"):
                    resume(root / "confirmation")
            path = root / "confirmation/cases/seed-4-data-2/plan.json"
            original = path.read_bytes()
            changed = json.loads(original)
            changed["corpus"]["data_seed"] = 0
            path.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, "case plan changed"):
                resume(root / "confirmation")
            path.write_bytes(original)
            with (root / "confirmation/.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaisesRegex(RuntimeError, "Another process"):
                    run_confirmation(root / "confirmation", plan)

    def test_all_six_real_negative_repeats_complete_resume_and_archive_offline(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            calibration = calibration_fixture(root, hypothetical_passing=True)
            stage = root / "confirmation"
            plan = freeze(stage, calibration, "short-lr001", render=False)
            calls = 0

            def interrupted(case, manifest, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 3:
                    raise RuntimeError("Simulated interruption between frozen cases")
                return run_stage(case, manifest, **kwargs)

            with patch("experiments.synthetic_trainers.stability_confirmation.run_stage", side_effect=interrupted):
                with self.assertRaisesRegex(RuntimeError, "Simulated interruption"):
                    run_confirmation(stage, plan)
            failed = json.loads((stage / "state.json").read_text())
            self.assertEqual((failed["status"], failed["completed_runs"]), ("failed", 2))
            initial_paths = [stage / "cases" / r["name"] / r["name"] / "measurements.json" for r in plan["recipes"][:2]]
            initial_bytes = [p.read_bytes() for p in initial_paths]
            result = run_confirmation(stage, resume(stage))
            self.assertEqual([p.read_bytes() for p in initial_paths], initial_bytes)
            self.assertEqual(result["completed_runs"], 6)
            self.assertFalse(result["repeatable_stable_benchmark"])
            raw_paths = [stage / "cases" / r["name"] / r["name"] / "measurements.json" for r in plan["recipes"]]
            originals = [p.read_bytes() for p in raw_paths]
            runs = {r["name"]: verify_archive(stage / "archives" / r["name"]) for r in plan["recipes"]}
            self.assertTrue(all(s["final"]["step"] == 8 for s in runs.values()))
            self.assertTrue(all(s["scope"].startswith("one_completed_independent_confirmation_run") for s in runs.values()))
            self.assertEqual(run_confirmation(stage, resume(stage)), result)
            self.assertEqual([p.read_bytes() for p in raw_paths], originals)
            name = plan["recipes"][-1]["name"]
            (stage / "archives" / name).rename(stage / "last-case")
            with self.assertRaises(FileNotFoundError):
                assemble(stage, root / "incomplete", render=False)
            self.assertFalse((root / "incomplete").exists())
            (stage / "last-case").rename(stage / "archives" / name)
            summary = assemble(stage, root / "portable", render=False)
            self.assertEqual(summary["total_updates"], 48)
            self.assertEqual(summary["completed_runs"], 6)
            self.assertFalse(summary["architecture_comparison_ready"])
            self.assertEqual(summary["long_confirmation_training_seconds"]["support"], 0)
            self.assertIsNone(summary["long_confirmation_training_seconds"]["mean"])
            hypothetical = deepcopy(runs)
            for run in hypothetical.values():
                run["assessment"]["stable_grokking"] = True
            self.assertTrue(derive(plan, calibration, hypothetical)["repeatable_stable_benchmark"])
            hypothetical[name]["assessment"]["stable_grokking"] = False
            self.assertFalse(derive(plan, calibration, hypothetical)["repeatable_stable_benchmark"])
            shutil.rmtree(stage)
            shutil.rmtree(calibration)
            self.assertEqual(verify_confirmation(root / "portable"), summary)
            subprocess.run([sys.executable, "-c", "import sys; "
                "from experiments.synthetic_trainers.confirmation_report import verify_confirmation; "
                "verify_confirmation(sys.argv[1]); assert 'torch' not in sys.modules", str(root / "portable")],
                check=True, capture_output=True, text=True)
            csv_path = root / "portable/confirmation.csv"
            csv_path.write_text("\n".join(csv_path.read_text().splitlines()[:-1]) + "\n")
            (root / "portable/artifact-hashes.json").write_text(json.dumps({"files": hashes(root / "portable")}))
            with self.assertRaisesRegex(ValueError, "confirmation.csv"):
                verify_confirmation(root / "portable")
