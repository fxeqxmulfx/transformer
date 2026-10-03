"""Actual six-case native tagged pipelines, fresh cohorts and retained negative outcomes."""

from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import torch

from experiments.synthetic_trainers import tagged_confirmation_protocol as protocol
from experiments.synthetic_trainers.paper_reproduction.provenance import ROOT
from experiments.synthetic_trainers.stability_comparison import hashes
from experiments.synthetic_trainers.stability_report import csv_rows,write_json
from experiments.synthetic_trainers.tagged_confirmation_layout import case_manifest,validate_plan
from experiments.synthetic_trainers.tagged_confirmation_report import assemble,derive,verify_archive,verify_confirmation,verified_benchmark


PROTOCOL=ROOT/"experiments/synthetic_trainers/protocols/adamw_stability_20261002"


class TaggedConfirmationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def freeze(self,stage,mode):
        path,name=("attention-pair-preparation/pair","adamw-softmax") if mode=="normalizer" else (
            "scheduled-pair-preparation/pair","adamw-cosine-tail")
        return protocol.freeze(stage,PROTOCOL/path,name,CPU_fixture=True,render=False)

    def test_scientific_selection_rejects_negative_CPU_references_and_sparsemax_primary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            for mode,path,name in (("normalizer","attention-pair-preparation/pair","adamw-softmax"),
                                    ("schedule","scheduled-pair-preparation/pair","adamw-cosine-tail")):
                with self.subTest(mode=mode),self.assertRaisesRegex(ValueError,"passing primary"):
                    protocol.freeze(root/mode,PROTOCOL/path,name,render=False)
                self.assertFalse((root/mode).exists())
            with self.assertRaisesRegex(ValueError,"primary benchmark retains softmax"):
                protocol.freeze(root/"sparse",PROTOCOL/"attention-pair-preparation/pair","adamw-sparsemax",CPU_fixture=True,render=False)
        self.assertFalse(torch.cuda.is_initialized())

    def test_exact_fresh_six_case_matrix_sources_tags_cap_and_seed_exclusions(self):
        with tempfile.TemporaryDirectory() as temporary:
            stage=Path(temporary)/"stage";plan=self.freeze(stage,"schedule")
            expected=[(4,2),(5,2),(6,2),(4,3),(5,3),(6,3)]
            self.assertEqual([(r["config"]["seed"],r["config"]["data_seed"]) for r in plan["recipes"]],expected)
            self.assertTrue(all(r["config"]["learning_rate_schedule"]=="cosine_tail" for r in plan["recipes"]))
            self.assertEqual(len({r["corpus"]["train_fingerprint"] for r in plan["recipes"]}),2)
            mutations=[]
            def change(action):
                modified=deepcopy(plan);action(modified);mutations.append(modified)
            change(lambda p:p["recipes"].pop())
            change(lambda p:p["model_seeds"].__setitem__(0,0))
            change(lambda p:p["data_seeds"].__setitem__(0,1))
            change(lambda p:p["recipes"][0]["config"].update(final_rate_factor=.2))
            change(lambda p:p["recipes"][0]["config"].update(attention_normalization="sparsemax"))
            change(lambda p:p.update(maximum_updates_per_run=600000))
            change(lambda p:p.update(scientific_run=True))
            change(lambda p:p.update(CPU_fixture=False))
            change(lambda p:p["source_hashes"].pop("experiments/synthetic_trainers/tagged_confirmation_protocol.py"))
            change(lambda p:p["calibration_complete_outcome"]["cases"]["adamw-cosine-tail"].update(stable_grokking=True))
            for index,modified in enumerate(mutations):
                with self.subTest(index=index),self.assertRaises(ValueError):
                    validate_plan(modified,stage/"calibration")
            with patch.object(protocol,"current_sources",return_value={}):
                with self.assertRaisesRegex(ValueError,"plan or sources changed"):
                    protocol.run_confirmation(stage,plan)
            self.assertFalse((stage/"state.json").exists())
            case=stage/"cases"/plan["recipes"][0]["name"]/"plan.json"
            altered=json.loads(case.read_text());altered["recipes"][0]["config"]["seed"]=0
            write_json(case,altered)
            with self.assertRaisesRegex(ValueError,"case plan changed"):
                protocol.run_confirmation(stage,plan)
            self.assertFalse((stage/"state.json").exists())

    def test_all_six_real_scheduled_negatives_archive_without_torch_and_rehashed_forgery_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);stage=root/"stage";plan=self.freeze(stage,"schedule")
            state=protocol.run_confirmation(stage,plan)
            self.assertEqual(state["completed_runs"],6);self.assertFalse(state["repeatable_stable_benchmark"])
            self.assertTrue(all(not row["assessment"]["stable_grokking"] for row in state["runs"]))
            with patch.object(protocol,"schedule_train",side_effect=AssertionError("Completed native cases must not be retrained")):
                self.assertEqual(protocol.run_confirmation(stage,plan),state)
            portable=root/"portable";summary=assemble(stage,portable,render=False)
            self.assertEqual(summary["total_updates"],240);self.assertEqual(summary["completed_runs"],6)
            self.assertFalse(summary["scientific_run"]);self.assertFalse(summary["architecture_comparison_ready"])
            # Hypothetical summary flags check the CPU scope, not a learning effect.
            hypothetical={r["name"]:verify_archive(stage/"archives"/r["name"]) for r in plan["recipes"]}
            for run in hypothetical.values():
                run["assessment"]["stable_grokking"]=True
            scoped=derive(plan,stage/"calibration",hypothetical)
            self.assertEqual(scoped["stable_grokking_runs"],6)
            self.assertFalse(scoped["repeatable_stable_benchmark"]);self.assertFalse(scoped["architecture_comparison_ready"])
            with self.assertRaisesRegex(ValueError,"all six scientific"):
                verified_benchmark(portable)
            last=plan["recipes"][-1]["name"];(portable/"runs"/last).rename(root/"missing")
            write_json(portable/"artifact-hashes.json",{"files":hashes(portable)})
            with self.assertRaises(FileNotFoundError):
                verify_confirmation(portable)
            (root/"missing").rename(portable/"runs"/last)
            write_json(portable/"artifact-hashes.json",{"files":hashes(portable)})
            shutil.rmtree(stage)
            self.assertEqual(verify_confirmation(portable),summary)
            code="from experiments.synthetic_trainers.tagged_confirmation_report import verify_confirmation; import sys,json; r=verify_confirmation(sys.argv[1]); print(json.dumps({'cases':r['completed_runs'],'scientific':r['scientific_run'],'torch_imported':'torch' in sys.modules}))"
            offline=json.loads(subprocess.check_output(["/usr/bin/python3","-c",code,str(portable)],text=True))
            self.assertEqual(offline,{"cases":6,"scientific":False,"torch_imported":False})
            metrics=json.loads((portable/"confirmation-recovery.json").read_text())
            metrics["recovery"][last]["tail_and_episodes"]["metrics"]["joint"]["episode_count"]=999
            write_json(portable/"confirmation-recovery.json",metrics)
            write_json(portable/"artifact-hashes.json",{"files":hashes(portable)})
            with self.assertRaisesRegex(ValueError,"recovery differs"):
                verify_confirmation(portable)

    def test_softmax_native_interruption_resume_and_byte_exact_gradient_archives(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);stage=root/"stage";plan=self.freeze(stage,"normalizer");count=0
            def stop_after_two(row):
                nonlocal count
                if row.get("event")=="completed_confirmation_case":
                    count+=1
                    if count==2:
                        raise RuntimeError("Injected interruption after two complete cases")
            with self.assertRaisesRegex(RuntimeError,"after two"):
                protocol.run_confirmation(stage,plan,progress=stop_after_two)
            state=json.loads((stage/"state.json").read_text())
            self.assertEqual((state["status"],state["completed_runs"]),("failed",2))
            first_hashes=[hashes(stage/"archives"/r["name"]) for r in plan["recipes"][:2]]
            self.assertEqual(protocol.run_confirmation(stage,plan)["completed_runs"],6)
            self.assertEqual([hashes(stage/"archives"/r["name"]) for r in plan["recipes"][:2]],first_hashes)
            first=plan["recipes"][0]["name"];archive=stage/"archives"/first
            summary=verify_archive(archive)
            gradients=[json.loads(line) for line in (archive/"gradients.jsonl").read_text().splitlines()]
            sampled={json.loads(line)["step"] for line in (archive/"diagnostics.jsonl").read_text().splitlines()}
            row=next(g for g in gradients if g["step"] not in sampled and g["step"]!=summary["gradient_trace"]["maximum_step"])
            row["gradient_l2"]*=.99
            (archive/"gradients.jsonl").write_text("".join(json.dumps(g)+"\n" for g in gradients))
            csv_rows(archive/"gradient-metrics.csv",gradients)
            write_json(archive/"artifact-hashes.json",{"files":hashes(archive)})
            verify_archive(archive)
            with self.assertRaisesRegex(ValueError,"complete raw records"):
                protocol.run_confirmation(stage,plan)
            self.assertEqual(json.loads((stage/"state.json").read_text())["status"],"failed")
