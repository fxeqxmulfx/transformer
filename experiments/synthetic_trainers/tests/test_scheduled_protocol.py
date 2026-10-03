"""Actual paired schedule archives, frozen scientific gates, and retained partial failures."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import torch

from experiments.synthetic_trainers import scheduled_protocol as protocol
from experiments.synthetic_trainers.paper_reproduction.diagnostics import DiagnosticsConfig
from experiments.synthetic_trainers.persistence import PersistenceConfig
from experiments.synthetic_trainers.runtime import write_json
from experiments.synthetic_trainers.scheduled_layout import validate_pair_plan
from experiments.synthetic_trainers.scheduled_pair_report import save_pair,verify_pair
from experiments.synthetic_trainers.scheduled_report import save_run,verify_archive
from experiments.synthetic_trainers.scheduled_training import ScheduledRunConfig
from experiments.synthetic_trainers.stability_comparison import hashes


class ScheduledProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def freeze(self,directory):
        config=ScheduledRunConfig(model="gptmini",optimizer="adamw",prime=7,train_fraction=.5,
            width=128,layers=2,heads=4,steps=40,batch_size=4,eval_every=5,
            learning_rate=.0003,weight_decay=.1,warmup_steps=10,device="cpu",
            anneal_start=20,anneal_end=30,final_rate_factor=.1)
        return protocol.freeze_pair(directory,config,DiagnosticsConfig(5,True,True),
            PersistenceConfig(plateau_steps=5,plateau_observations=2,confirmation_observations=2,tail_steps=10))

    def test_real_negative_pair_archives_without_torch_and_completed_cases_are_not_retrained(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);stage=root/"stage";plan=self.freeze(stage)
            state=protocol.run_pair(stage,plan)
            self.assertEqual(state["completed_runs"],2)
            self.assertEqual([row["name"] for row in state["runs"]],plan["execution_order"])
            self.assertTrue(all(not row["assessment"]["stable_grokking"] for row in state["runs"]))
            with patch.object(protocol,"train",side_effect=AssertionError("Completed cases cannot be retrained")):
                self.assertEqual(protocol.run_pair(stage,plan),state)
            archives=[]
            for recipe in plan["recipes"]:
                destination=root/recipe["name"]
                save_run(stage,recipe["name"],destination,render=False)
                archives.append(destination)
            pair=root/"pair";result=save_pair(archives,pair,render=False)
            self.assertFalse(result["scientific_run"])
            self.assertEqual(result["eligible_pair_support"],0)
            self.assertIsNone(result["control_to_candidate_training_time_ratio"])
            self.assertEqual(result["eligible_scientific_calibration_recipes"],[])
            self.assertFalse(result["campaign_architecture_claim_allowed"])
            self.assertEqual(verify_pair(pair),result)
            code="from experiments.synthetic_trainers.scheduled_pair_report import verify_pair; import sys,json; r=verify_pair(sys.argv[1]); print(json.dumps({'complete_pair':r['complete_pair'],'torch_imported':'torch' in sys.modules}))"
            offline=json.loads(subprocess.check_output(["/usr/bin/python3","-c",code,str(pair)],text=True))
            self.assertEqual(offline,{"complete_pair":True,"torch_imported":False})
            forged=deepcopy(result)
            forged["recovery"]["adamw-cosine-tail"]["tail_and_episodes"]["metrics"]["joint"]["episode_count"]=999
            write_json(pair/"schedule-pair.json",forged)
            write_json(pair/"artifact-hashes.json",{"files":hashes(pair)})
            with self.assertRaisesRegex(ValueError,"frequency/recovery"):
                verify_pair(pair)

    def test_other_interventions_dropped_cases_changed_grids_caps_and_source_claims_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            plan=self.freeze(Path(temporary)/"stage")
            mutations=[]
            def change(action):
                changed=deepcopy(plan);action(changed);mutations.append(changed)
            change(lambda p:p["recipes"][1]["config"].update(learning_rate=.001))
            change(lambda p:p["recipes"].pop())
            change(lambda p:p["execution_order"].reverse())
            change(lambda p:p["recipes"][1]["config"].update(learning_rate_schedule="constant"))
            change(lambda p:p.update(recovery_window_steps=5000))
            change(lambda p:p["final_window"].__setitem__(0,35))
            change(lambda p:p.update(maximum_updates_per_run=600000))
            change(lambda p:p.update(campaign_architecture_claim_allowed=True))
            change(lambda p:p["training_source_hashes"].pop("experiments/synthetic_trainers/scheduled_rates.py"))
            change(lambda p:p["source_hashes"].pop("experiments/synthetic_trainers/scheduled_pair_report.py"))
            change(lambda p:p["recipes"][1]["config"].update(final_rate_factor=.2))
            change(lambda p:p.update(scientific_run=True))
            for index,changed in enumerate(mutations):
                with self.subTest(index=index),self.assertRaises(ValueError):
                    validate_pair_plan(changed)

    def test_preflight_rejects_drift_and_completed_first_case_survives_later_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);stage=root/"stage";plan=self.freeze(stage)
            changed=deepcopy(plan);changed["recipes"][1]["config"]["seed"]+=1
            write_json(stage/"plan.json",changed)
            with patch.object(protocol,"train",side_effect=AssertionError("Must reject before training")):
                with self.assertRaisesRegex(ValueError,"plan changed"):
                    protocol.run_pair(stage,plan)
            write_json(stage/"plan.json",plan)
            altered=deepcopy(plan["source_hashes"]);altered["experiments/synthetic_trainers/scheduled_rates.py"]="0"*64
            with patch.object(protocol,"current_sources",return_value=altered):
                with self.assertRaisesRegex(ValueError,"sources changed"):
                    protocol.run_pair(stage,plan)
            self.assertFalse((stage/"state.json").exists())
            original=protocol.verify_live;calls=0
            def reject_second(directory,expected):
                nonlocal calls
                calls+=1
                if calls==3:
                    raise ValueError("Injected drift before second case")
                return original(directory,expected)
            with patch.object(protocol,"verify_live",side_effect=reject_second):
                with self.assertRaisesRegex(ValueError,"before second case"):
                    protocol.run_pair(stage,plan)
            state=json.loads((stage/"state.json").read_text())
            self.assertEqual(state["status"],"failed");self.assertEqual(state["completed_runs"],1)
            archive=root/"constant";save_run(stage,"adamw-constant",archive,render=False)
            self.assertTrue(verify_archive(archive)["complete_run"])
            self.assertFalse((stage/"adamw-cosine-tail").exists())

    def test_scientific_selection_waits_for_complete_failed_reference_before_cuda_initialization(self):
        config=ScheduledRunConfig(model="gptmini",optimizer="adamw",steps=300000)
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            with self.assertRaisesRegex(ValueError,"complete normalizer"):
                protocol.freeze_pair(root/"missing",config,DiagnosticsConfig(),PersistenceConfig())
            passing={"complete_pair":True,"scientific_run":True,"outcome":{"control":{"stable_grokking":True}}}
            with patch.object(protocol,"verify_reference",return_value=passing):
                with self.assertRaisesRegex(ValueError,"independent confirmation"):
                    protocol.freeze_pair(root/"passing",config,DiagnosticsConfig(),PersistenceConfig(),reference=root)
            self.assertFalse((root/"missing").exists());self.assertFalse((root/"passing").exists())
        self.assertFalse(torch.cuda.is_initialized())
