"""Actual CPU paired archives, immutable controls and failures after rehashing."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import torch

from experiments.synthetic_trainers import attention_protocol as protocol
from experiments.synthetic_trainers.attention_layout import validate_pair_plan
from experiments.synthetic_trainers.attention_report import save_pair,verify_pair
from experiments.synthetic_trainers.attention_training import AttentionRunConfig
from experiments.synthetic_trainers.paper_reproduction.diagnostics import DiagnosticsConfig
from experiments.synthetic_trainers.persistence import PersistenceConfig
from experiments.synthetic_trainers.runtime import write_json
from experiments.synthetic_trainers.stability_comparison import hashes
from experiments.synthetic_trainers.stability_report import save_run


class AttentionProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def freeze(self,directory):
        config=AttentionRunConfig(model="gptmini",optimizer="adamw",prime=7,train_fraction=.5,
            width=128,layers=2,heads=4,steps=20,batch_size=4,eval_every=5,
            learning_rate=.0003,weight_decay=.1,warmup_steps=10,device="cpu")
        return protocol.freeze_pair(directory,config,DiagnosticsConfig(5,True,True),
            PersistenceConfig(plateau_steps=5,plateau_observations=2,confirmation_observations=2,tail_steps=10))

    def test_complete_real_negative_pair_archives_portably_and_completed_cases_are_not_retrained(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);stage=root/"stage"
            plan=self.freeze(stage)
            state=protocol.run_pair(stage,plan)
            self.assertEqual(state["completed_runs"],2)
            self.assertEqual([row["name"] for row in state["runs"]],plan["execution_order"])
            self.assertTrue(all(not row["assessment"]["stable_grokking"] for row in state["runs"]))
            with patch.object(protocol,"train",side_effect=AssertionError("Completed cases must not be retrained")):
                self.assertEqual(protocol.run_pair(stage,plan),state)
            paths=[]
            for recipe in plan["recipes"]:
                destination=root/recipe["name"]
                save_run(stage,recipe["name"],destination,render=False)
                paths.append(destination)
            pair=root/"pair"
            result=save_pair(paths,pair,render=False)
            self.assertFalse(result["scientific_run"])
            self.assertEqual(result["outcome"]["eligible_pair_support"],0)
            self.assertIsNone(result["outcome"]["control_to_candidate_training_time_ratio"])
            self.assertFalse(result["campaign_architecture_claim_allowed"])
            self.assertEqual(verify_pair(pair),result)
            code="from experiments.synthetic_trainers.attention_report import verify_pair; import sys,json; result=verify_pair(sys.argv[1]); print(json.dumps({'complete_pair':result['complete_pair'],'torch_imported':'torch' in sys.modules}))"
            offline=json.loads(subprocess.check_output(["/usr/bin/python3","-c",code,str(pair)],text=True))
            self.assertEqual(offline,{"complete_pair":True,"torch_imported":False})
            forged=deepcopy(result)
            forged["recovery"]["adamw-sparsemax"]["tail_and_episodes"]["metrics"]["joint"]["episode_count"]=999
            write_json(pair/"normalizer-pair.json",forged)
            write_json(pair/"artifact-hashes.json",{"files":hashes(pair)})
            with self.assertRaisesRegex(ValueError,"frequency/recovery"):
                verify_pair(pair)

    def test_invalid_interventions_dropped_cases_changed_grids_and_caps_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            plan=self.freeze(Path(temporary)/"stage")
            mutations=[]
            changed=deepcopy(plan);changed["recipes"][1]["config"]["learning_rate"]*=2;mutations.append(changed)
            changed=deepcopy(plan);changed["recipes"].pop();mutations.append(changed)
            changed=deepcopy(plan);changed["recipes"][1]["config"]["attention_normalization"]="softmax";mutations.append(changed)
            changed=deepcopy(plan);changed["recovery_window_steps"]=5000;mutations.append(changed)
            changed=deepcopy(plan);changed["final_window"][0]+=5;mutations.append(changed)
            changed=deepcopy(plan);changed["maximum_updates_per_run"]=600000;mutations.append(changed)
            changed=deepcopy(plan);changed["campaign_architecture_claim_allowed"]=True;mutations.append(changed)
            changed=deepcopy(plan);del changed["training_source_hashes"]["experiments/synthetic_trainers/sparsemax_attention.py"];mutations.append(changed)
            changed=deepcopy(plan);del changed["source_hashes"]["experiments/synthetic_trainers/attention_report.py"];mutations.append(changed)
            changed=deepcopy(plan);changed["scientific_run"]=True;mutations.append(changed)
            changed=deepcopy(plan);changed["improvement_rule"]["minimum_time_ratio"]=1;mutations.append(changed)
            changed=deepcopy(plan);changed["execution_order"].reverse();mutations.append(changed)
            for index,changed in enumerate(mutations):
                with self.subTest(index=index):
                    with self.assertRaises(ValueError):
                        validate_pair_plan(changed)

    def test_changed_frozen_plan_and_sources_are_rejected_before_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            stage=Path(temporary)/"stage";plan=self.freeze(stage)
            changed=deepcopy(plan);changed["recipes"][1]["config"]["seed"]+=1
            write_json(stage/"plan.json",changed)
            with patch.object(protocol,"train",side_effect=AssertionError("Must reject before training")):
                with self.assertRaisesRegex(ValueError,"plan changed"):
                    protocol.run_pair(stage,plan)
            write_json(stage/"plan.json",plan)
            altered=deepcopy(plan["source_hashes"])
            altered["experiments/synthetic_trainers/sparsemax_attention.py"]="0"*64
            with patch.object(protocol,"current_sources",return_value=altered):
                with patch.object(protocol,"train",side_effect=AssertionError("Must reject before training")):
                    with self.assertRaisesRegex(ValueError,"sources changed"):
                        protocol.run_pair(stage,plan)
            self.assertFalse((stage/"state.json").exists())

    def test_scientific_freeze_waits_for_complete_reference_before_cuda_initialization(self):
        config=AttentionRunConfig(model="gptmini",optimizer="adamw",steps=300000)
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError,"completed reference"):
                protocol.freeze_pair(Path(temporary)/"stage",config,DiagnosticsConfig(),PersistenceConfig())
        self.assertFalse(torch.cuda.is_initialized())
