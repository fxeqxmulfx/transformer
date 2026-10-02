"""Original constant trajectory, fixed annealing rates and native continuation."""

from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from experiments.synthetic_trainers import scheduled_training as scheduled
from experiments.synthetic_trainers.paper_reproduction import grokking
from experiments.synthetic_trainers.paper_reproduction.diagnostics import DiagnosticsConfig


class ScheduledTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def config(self, **kwargs):
        return scheduled.ScheduledRunConfig(**(dict(model="gptmini", optimizer="adamw", prime=7,
            train_fraction=.5, steps=40, batch_size=4, eval_every=5, learning_rate=.0003,
            weight_decay=.1, anneal_start=20, anneal_end=30, device="cpu") | kwargs))

    def equal(self, left, right):
        if isinstance(left, torch.Tensor):
            self.assertTrue(torch.equal(left, right))
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.equal(a, b)
        else:
            self.assertEqual(left, right)

    def checkpoints(self, left, right):
        for key in ("model", "optimizer", "step", "examples_seen", "last_batch_size",
                    "batch_generator_state", "permutation", "cursor"):
            self.equal(left[key], right[key])

    def history(self, left, right):
        for a, b in zip(left["history"], right["history"]):
            self.assertEqual({k:v for k,v in a.items() if k not in ("training_seconds", "wall_seconds")},
                             {k:v for k,v in b.items() if k not in ("training_seconds", "wall_seconds")})
        self.assertEqual(len(left["history"]), len(right["history"]))

    def test_constant_policy_retains_the_original_full_width_trajectory_and_actual_rates(self):
        config = self.config()
        plain = grokking.RunConfig(**{k:v for k,v in asdict(config).items()
                                    if k not in ("learning_rate_schedule", "anneal_start", "anneal_end", "final_rate_factor")})
        diagnostics = DiagnosticsConfig(5, True, True)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);base=root/"plain";wrapped=root/"constant"
            expected=grokking.train(plain,base,diagnostics=diagnostics)
            actual=scheduled.train(config,wrapped,diagnostics=diagnostics)
            self.checkpoints(torch.load(base/"checkpoint.pt",weights_only=True),
                             torch.load(wrapped/"checkpoint.pt",weights_only=True))
            self.history(expected,actual)
            for name in ("gradients.jsonl", "diagnostics.jsonl"):
                self.assertEqual((base/name).read_bytes(),(wrapped/name).read_bytes())
            self.assertEqual(actual["plan"]["source_hashes"],scheduled.training_sources())
            self.assertEqual(actual["plan"]["config"]["learning_rate_schedule"],"constant")

    def test_annealing_preserves_prefix_and_resume_with_native_buffers_sampling_and_rates(self):
        config=self.config(learning_rate_schedule="cosine_tail")
        diagnostics=DiagnosticsConfig(5,True,True)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);constant=root/"constant";fresh=root/"fresh";continued=root/"continued"
            scheduled.train(replace(config,learning_rate_schedule="constant"),constant,diagnostics=diagnostics)
            expected=scheduled.train(config,fresh,diagnostics=diagnostics)
            scheduled.train(replace(config,steps=30),continued,diagnostics=diagnostics)
            actual=scheduled.train(config,continued,resume=True,diagnostics=diagnostics)
            self.checkpoints(torch.load(fresh/"checkpoint.pt",weights_only=True),
                             torch.load(continued/"checkpoint.pt",weights_only=True))
            self.history(expected,actual)
            for name in ("gradients.jsonl", "diagnostics.jsonl"):
                self.assertEqual((fresh/name).read_bytes(),(continued/name).read_bytes())
            fixed=[json.loads(line) for line in (constant/"gradients.jsonl").read_text().splitlines()]
            annealed=[json.loads(line) for line in (fresh/"gradients.jsonl").read_text().splitlines()]
            self.assertEqual(fixed[:21],annealed[:21])
            self.assertEqual(annealed[0]["learning_rate"],0)
            self.assertAlmostEqual(annealed[25]["learning_rate"],.000165,places=14)
            self.assertTrue(all(abs(row["learning_rate"]-.00003)<1e-14 for row in annealed[30:]))
            self.assertTrue(all(a["learning_rate"]>=b["learning_rate"] for a,b in zip(annealed[20:],annealed[21:])))
            native=torch.load(continued/"checkpoint.pt",weights_only=True)["optimizer"]
            self.assertTrue(all(int(s["step"])==40 for s in native["state"].values()))

    def test_changed_schedule_resume_invalid_intervals_and_exceptions_fail_without_mutation(self):
        config=self.config(steps=30,learning_rate_schedule="cosine_tail")
        rate,hashes=grokking.learning_rate,grokking.source_hashes
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp)/"case"
            scheduled.train(config,directory)
            before={p.name:p.read_bytes() for p in directory.iterdir() if p.is_file()}
            with patch.object(grokking.torch,"load",side_effect=AssertionError("Reject before loading")):
                for changed in (replace(config,learning_rate_schedule="constant"),replace(config,final_rate_factor=.2)):
                    with self.assertRaisesRegex(ValueError,"original update budget"):
                        scheduled.train(changed,directory,resume=True)
            self.assertEqual(before,{p.name:p.read_bytes() for p in directory.iterdir() if p.is_file()})
            with patch.object(grokking,"evaluate",side_effect=RuntimeError("Interrupted fixture")):
                with self.assertRaisesRegex(RuntimeError,"Interrupted fixture"):
                    scheduled.train(config,Path(tmp)/"failed")
        self.assertIs(grokking.learning_rate,rate);self.assertIs(grokking.source_hashes,hashes)
        for change in ({"learning_rate_schedule":"adaptive"},{"steps":300001},{"anneal_start":5},
                       {"anneal_end":41},{"final_rate_factor":0},{"final_rate_factor":1},
                       {"final_rate_factor":float("nan")},{"optimizer":"amsgradw"},{"model":"reference"}):
            with self.assertRaises(ValueError): self.config(**change)
        with self.assertRaises(TypeError): scheduled.train(grokking.RunConfig(),"unused")
