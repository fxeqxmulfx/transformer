"""Full real native pair matrices, scientific gates and portable negative reports."""

from dataclasses import replace
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import torch

from experiments.synthetic_trainers import architecture_cohort_protocol as protocol
from experiments.synthetic_trainers.architecture_cohort_layout import validate_plan
from experiments.synthetic_trainers.architecture_cohort_report import assemble, verify_case, verify_cohort
from experiments.synthetic_trainers.architecture_training import ArchitectureRunConfig, make_model
from experiments.synthetic_trainers.paper_reproduction.provenance import ROOT
from experiments.synthetic_trainers.stability_comparison import hashes
from experiments.synthetic_trainers.stability_report import csv_rows, write_json

PROTOCOL = ROOT / "experiments/synthetic_trainers/protocols/adamw_stability_20261002"


class ArchitectureCohortTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def freeze(self, stage, mode):
        return protocol.freeze(stage, PROTOCOL / "tagged-confirmation-preparation" / mode,
                               CPU_fixture=True, render=False)

    def test_scientific_gate_rejects_actual_negative_six_case_and_incomplete_pair_references(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for mode in ("normalizer", "schedule"):
                with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "all six scientific"):
                    protocol.freeze(root / mode, PROTOCOL / "tagged-confirmation-preparation" / mode)
                self.assertFalse((root / mode).exists())
            with self.assertRaisesRegex(ValueError, "all six scientific"):
                protocol.freeze(root / "pair", PROTOCOL / "attention-pair-preparation/pair")
            self.assertFalse((root / "pair").exists())
        self.assertFalse(torch.cuda.is_initialized())

    def test_fresh_matrix_native_schedule_pair_initialization_and_preflight_rejections(self):
        with tempfile.TemporaryDirectory() as temporary:
            stage = Path(temporary) / "stage"; plan = self.freeze(stage, "schedule")
            self.assertEqual([(p["model_seed"], p["data_seed"]) for p in plan["pairs"]],
                             [(7, 4), (8, 4), (9, 4), (7, 5), (8, 5), (9, 5)])
            self.assertEqual(len(plan["training_source_hashes"]), 19)
            self.assertTrue(all(r["config"]["learning_rate_schedule"] == "cosine_tail" for r in plan["recipes"]))
            self.assertEqual(len({r["corpus"]["train_fingerprint"] for r in plan["recipes"]}), 2)
            for seed in (7, 8, 9):
                config = ArchitectureRunConfig(**{**plan["base_config"], "seed": seed})
                left = make_model(config, 149); rng = torch.get_rng_state().clone()
                right = make_model(replace(config, attention_normalization="sparsemax"), 149)
                self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                self.assertTrue(all(torch.equal(v, right.state_dict()[k]) for k, v in left.state_dict().items()))
            mutations = []
            def change(action):
                p = json.loads(json.dumps(plan)); action(p); mutations.append(p)
            change(lambda p: p.update(planned_pairs=5))
            change(lambda p: p["recipes"].pop())
            change(lambda p: p["model_seeds"].__setitem__(0, 4))
            change(lambda p: p["data_seeds"].__setitem__(0, 2))
            change(lambda p: p["recipes"][1]["config"].update(learning_rate=.001))
            change(lambda p: p["recipes"][1]["config"].update(final_rate_factor=.2))
            change(lambda p: p["recipes"][1]["corpus"].update(train_fingerprint="forged"))
            change(lambda p: p["execution_order"].reverse())
            change(lambda p: p["source_hashes"].pop("experiments/synthetic_trainers/architecture_cohort_protocol.py"))
            change(lambda p: p["training_source_hashes"].pop("experiments/synthetic_trainers/architecture_training.py"))
            change(lambda p: p["improvement_rule"].update(minimum_time_ratio=.5))
            change(lambda p: p.update(maximum_updates_per_run=600000))
            change(lambda p: p.update(initialization_pairing="unverified"))
            change(lambda p: p.update(scientific_run=True))
            for index, modified in enumerate(mutations):
                with self.subTest(index=index), self.assertRaises(ValueError):
                    validate_plan(modified, stage / "benchmark")
            with patch.object(protocol, "current_sources", return_value={}):
                with self.assertRaisesRegex(ValueError, "plan or sources changed"):
                    protocol.run_cohort(stage, plan)
            self.assertFalse((stage / "state.json").exists())
            path = stage / "cases" / plan["recipes"][0]["name"] / "plan.json"
            case = json.loads(path.read_text()); case["recipes"][0]["config"]["seed"] = 0
            write_json(path, case)
            with self.assertRaisesRegex(ValueError, "case plan changed"):
                protocol.run_cohort(stage, plan)
            self.assertFalse((stage / "state.json").exists())
        self.assertFalse(torch.cuda.is_initialized())

    def test_all_six_real_scheduled_pairs_remain_negative_portable_and_complete(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); stage = root / "stage"; plan = self.freeze(stage, "schedule")
            state = protocol.run_cohort(stage, plan)
            self.assertEqual(state["completed_runs"], 12)
            with patch.object(protocol, "train", side_effect=AssertionError("Completed native cases must not retrain")):
                self.assertEqual(protocol.run_cohort(stage, plan), state)
            portable = root / "portable"; result = assemble(stage, portable, render=False)
            self.assertEqual((result["completed_pairs"], result["total_updates"]), (6, 480))
            self.assertEqual(result["eligible_pair_support"], 0)
            self.assertFalse(result["repeatable_architecture_improvement"])
            self.assertFalse(result["scientific_primary_benchmark_gate_verified"])
            self.assertTrue(all(p["control_to_candidate_training_time_ratio"] is None for p in result["pairs"].values()))
            shutil.rmtree(stage)
            self.assertEqual(verify_cohort(portable), result)
            code = "from experiments.synthetic_trainers.architecture_cohort_report import verify_cohort; import sys,json; r=verify_cohort(sys.argv[1]); print(json.dumps({'pairs':r['completed_pairs'],'support':r['eligible_pair_support'],'torch_imported':'torch' in sys.modules}))"
            offline = json.loads(subprocess.check_output(["/usr/bin/python3", "-c", code, str(portable)], text=True))
            self.assertEqual(offline, {"pairs": 6, "support": 0, "torch_imported": False})
            last = plan["recipes"][-1]["name"]
            (portable / "runs" / last).rename(root / "removed")
            write_json(portable / "artifact-hashes.json", {"files": hashes(portable)})
            with self.assertRaisesRegex(ValueError, "including failures"):
                verify_cohort(portable)
            (root / "removed").rename(portable / "runs" / last)
            metrics = json.loads((portable / "architecture-recovery.json").read_text())
            metrics["recovery"][last]["tail_and_episodes"]["metrics"]["joint"]["episode_count"] = 999
            write_json(portable / "architecture-recovery.json", metrics)
            write_json(portable / "artifact-hashes.json", {"files": hashes(portable)})
            with self.assertRaisesRegex(ValueError, "recovery metrics differ"):
                verify_cohort(portable)

    def test_full_native_normalizer_interruption_resume_and_raw_archive_equality(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); stage = root / "stage"; plan = self.freeze(stage, "normalizer"); count = 0
            self.assertEqual(plan["base_config"]["anneal_start"], 10)
            self.assertEqual(plan["base_config"]["anneal_end"], 20)
            def interrupt(row):
                nonlocal count
                if row.get("event") == "completed_architecture_case":
                    count += 1
                    if count == 2:
                        raise RuntimeError("Injected interruption after two complete architecture cases")
            with self.assertRaisesRegex(RuntimeError, "after two"):
                protocol.run_cohort(stage, plan, progress=interrupt)
            state = json.loads((stage / "state.json").read_text())
            self.assertEqual((state["status"], state["completed_runs"]), ("failed", 2))
            preserved = [hashes(stage / "archives" / r["name"]) for r in plan["recipes"][:2]]
            self.assertEqual(protocol.run_cohort(stage, plan)["completed_runs"], 12)
            self.assertEqual([hashes(stage / "archives" / r["name"]) for r in plan["recipes"][:2]], preserved)
            archive = stage / "archives" / plan["recipes"][0]["name"]; summary = verify_case(archive)
            gradients = [json.loads(line) for line in (archive / "gradients.jsonl").read_text().splitlines()]
            sampled = {json.loads(line)["step"] for line in (archive / "diagnostics.jsonl").read_text().splitlines()}
            row = next(g for g in gradients if g["step"] not in sampled and g["step"] != summary["gradient_trace"]["maximum_step"])
            row["gradient_l2"] *= .99
            (archive / "gradients.jsonl").write_text("".join(json.dumps(g) + "\n" for g in gradients))
            csv_rows(archive / "gradient-metrics.csv", gradients)
            write_json(archive / "artifact-hashes.json", {"files": hashes(archive)})
            verify_case(archive)
            with self.assertRaisesRegex(ValueError, "complete raw records"):
                protocol.run_cohort(stage, plan)
            self.assertEqual(json.loads((stage / "state.json").read_text())["status"], "failed")


if __name__ == "__main__":
    unittest.main()
