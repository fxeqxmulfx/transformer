"""Actual native updates, recorded schedules, and unchanged study semantics."""

from dataclasses import replace
import json
import math
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import torch

from experiments.gpt_mini import GPTMini
from experiments.synthetic_trainers import complementary_training as adapter
from experiments.synthetic_trainers import training
from experiments.synthetic_trainers.complementary_config import ComplementaryConfig
from experiments.synthetic_trainers.complementary_report import hashes, verify_run
from experiments.synthetic_trainers.config import ModelSpec, TrainConfig
from experiments.synthetic_trainers.corpus import study_pool
from experiments.synthetic_trainers.metrics import masked_loss
from experiments.synthetic_trainers.paper_reproduction.grokking import RunConfig, make_optimizer
from experiments.synthetic_trainers.records import collate
from experiments.synthetic_trainers.runtime import write_json
from experiments.synthetic_trainers.specs import TaskSpec


def equal_state(test, left, right):
    if isinstance(left, torch.Tensor):
        test.assertTrue(torch.equal(left, right))
    elif isinstance(left, dict):
        test.assertEqual(left.keys(), right.keys())
        for key in left:
            equal_state(test, left[key], right[key])
    elif isinstance(left, (tuple, list)):
        test.assertEqual(len(left), len(right))
        for first, second in zip(left, right):
            equal_state(test, first, second)
    else:
        test.assertEqual(left, right)


class ComplementaryTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def config(self):
        return ComplementaryConfig(steps=16, eval_every=4, batch_size=2,
                                   train_examples=3, validation_examples=3, test_examples=3,
                                   eval_lengths=(8,), target=1.0)

    def test_actual_updates_match_independent_primary_optimizer_and_schedule(self):
        spec = TaskSpec(task="copy", length=4, symbols=8)
        model_spec = ModelSpec(width=8, layers=1, heads=1, init_std=.02)
        for schedule in ("constant", "cosine_tail"):
            cfg = replace(self.config(), learning_rate_schedule=schedule, anneal_start=10, anneal_end=14)
            with self.subTest(schedule=schedule), tempfile.TemporaryDirectory() as root:
                result = adapter.train(cfg, spec, model_spec, root)
                torch.manual_seed(cfg.seed)
                model = GPTMini(model_spec.reference_config(result["provenance"]["vocab_size"],
                                                            result["provenance"]["context_length"]))
                for parameter in model.parameters():
                    if parameter.ndim == 2:
                        torch.nn.init.normal_(parameter, std=.02)
                initial = {key: value.clone() for key, value in model.state_dict().items()}
                original = RunConfig(model="gptmini", device="cpu", steps=cfg.steps,
                                     learning_rate=cfg.learning_rate, weight_decay=cfg.weight_decay)
                optimizer = make_optimizer(model, original)
                rows = study_pool(spec, cfg)["train"].examples
                generator = torch.Generator().manual_seed(cfg.seed)
                order, cursor = [], 0
                expected_rates = []
                for step in range(1, cfg.steps + 1):
                    if cursor == len(order):
                        order = torch.randperm(len(rows), generator=generator).tolist()
                        cursor = 0
                    indices = order[cursor:cursor + cfg.batch_size]
                    cursor += len(indices)
                    batch = collate([rows[index] for index in indices], "cpu")
                    model.train()
                    optimizer.zero_grad(set_to_none=True)
                    masked_loss(model(batch.tokens), batch.targets).backward()
                    rate = .0003 * min(1, (step - 1) / 10)
                    if schedule == "cosine_tail" and step - 1 > 10:
                        fraction = min(1, (step - 1 - 10) / 4)
                        rate *= .1 + .9 * (1 + math.cos(math.pi * fraction)) / 2
                    for group in optimizer.param_groups:
                        group["lr"] = rate
                    optimizer.step()
                    expected_rates.append(rate)
                    if step == 1:
                        equal_state(self, initial, model.state_dict())
                        self.assertTrue(any(state["exp_avg"].abs().sum() > 0 for state in optimizer.state.values()))
                equal_state(self, model.state_dict(), torch.load(Path(root) / "final.pt", weights_only=True))
                equal_state(self, optimizer.state_dict(), torch.load(Path(root) / "optimizer-final.pt", weights_only=True))
                actual = [json.loads(line) for line in (Path(root) / "update-rates.jsonl").read_text().splitlines()]
                self.assertEqual([row["step"] for row in actual], list(range(1, 17)))
                self.assertEqual([row["learning_rates"] for row in actual], [[rate] for rate in expected_rates])
                self.assertEqual([row["weight_decays"] for row in actual], [[.1]] * 16)
                audit = result["complementary_protocol"]["optimizer_audit"]
                self.assertEqual(audit["parameter_scalars"], result["parameters"])
                self.assertEqual(audit["state_steps"], [16] * audit["parameter_tensors"])
                self.assertEqual(result["examples_seen"], 24)  # alternating full and short batches
                self.assertEqual(set(result["test_final"]), {"in_distribution", "length-8"})
                self.assertEqual(result["study"]["generalization_transition"]["novel_validation_examples"], 3)

    def test_invalid_protocol_is_rejected_before_any_model_or_output(self):
        for fields in ({"steps": 300001}, {"steps": 3.5}, {"optimizer": "amsgradw"},
                       {"beta2": .999}, {"grad_clip": 1}, {"decay_scope": "matrices"},
                       {"label_noise": .1}, {"split_policy": "independent"},
                       {"warmup_steps": -1}, {"learning_rate_schedule": "cosine_tail", "anneal_end": 17},
                       {"final_rate_factor": float("nan")}, {"learning_rate_schedule": "adaptive"}):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                replace(self.config(), **fields)
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "absent"
            with patch.object(training, "GPTMini", side_effect=AssertionError("model should not be constructed")):
                with self.assertRaises(TypeError):
                    adapter.train(TrainConfig(), TaskSpec(task="copy", length=4), ModelSpec(), path)
                with self.assertRaises(ValueError):
                    adapter.train(self.config(), TaskSpec(task="copy", length=4), ModelSpec(), path)
            self.assertFalse(path.exists())

    def test_scoped_adapter_restores_generic_factory_after_success_and_failure(self):
        spec = TaskSpec(task="copy", length=4, symbols=8)
        model_spec = ModelSpec(width=8, layers=1, heads=1, init_std=.02)
        optimizer_for, description = training.optimizer_for, training.optimizer_description
        with tempfile.TemporaryDirectory() as root:
            adapter.train(replace(self.config(), steps=1), spec, model_spec, Path(root) / "success")
            with self.assertRaisesRegex(RuntimeError, "intentional"):
                adapter.train(self.config(), spec, model_spec, Path(root) / "interrupted",
                              progress=lambda row: (_ for _ in ()).throw(RuntimeError("intentional")))
        self.assertIs(training.optimizer_for, optimizer_for)
        self.assertIs(training.optimizer_description, description)
        self.assertEqual(training.optimizer_description(TrainConfig())["decay_scope"], "matrices")

    def test_portable_archive_rejects_rehashed_rate_and_generated_data_forgeries(self):
        spec = TaskSpec(task="copy", length=4, symbols=8)
        model_spec = ModelSpec(width=8, layers=1, heads=1, init_std=.02)
        cfg = self.config()
        with tempfile.TemporaryDirectory() as root:
            adapter.train(cfg, spec, model_spec, root)
            directory = Path(root)
            write_json(directory / "artifact-hashes.json", {"files": hashes(directory)})
            self.assertEqual(verify_run(directory)["actual_rate_observations"], 16)
            script = ("import json,sys; from experiments.synthetic_trainers.complementary_report import verify_run; "
                      "print(json.dumps(verify_run(sys.argv[1]))); assert 'torch' not in sys.modules")
            subprocess.run(["python3", "-c", script, root], check=True, capture_output=True, text=True)
            path = directory / "update-rates.jsonl"
            original = path.read_bytes()
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            rows[4]["learning_rates"] = [.0003]
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            write_json(directory / "artifact-hashes.json", {"files": hashes(directory)})
            with self.assertRaisesRegex(ValueError, "learning-rate history"):
                verify_run(directory)
            path.write_bytes(original)
            path = directory / "data" / "validation" / "examples.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            rows[0]["tokens"][0] = 999
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            write_json(directory / "artifact-hashes.json", {"files": hashes(directory)})
            with self.assertRaisesRegex(ValueError, "deterministic generator"):
                verify_run(directory)
