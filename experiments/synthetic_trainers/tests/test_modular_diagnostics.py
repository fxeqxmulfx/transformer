"""Instrumentation must preserve updates, moment buffers, and resume behavior."""

from dataclasses import replace
import json
import math
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn

from experiments.optimizer_benchmark.coordinate import CoordinateOptimizer
from experiments.synthetic_trainers.paper_reproduction.diagnostics import (
    DiagnosticsConfig, after_update, before_update,
)
from experiments.synthetic_trainers.paper_reproduction.grokking import RunConfig, train
from experiments.synthetic_trainers.paper_reproduction.provenance import source_hashes
from experiments.synthetic_trainers.reproduction import run_campaign


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def assert_checkpoint_equal(test, left, right):
    test.assertEqual(left["step"], right["step"])
    test.assertEqual(left["examples_seen"], right["examples_seen"])
    test.assertEqual(left["cursor"], right["cursor"])
    test.assertEqual(left["optimizer_steps"], right["optimizer_steps"])
    for key in ("permutation", "batch_generator_state"):
        torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)
    for name, value in left["model"].items():
        torch.testing.assert_close(value, right["model"][name], rtol=0, atol=0)
    test.assertEqual(left["optimizer"]["param_groups"], right["optimizer"]["param_groups"])
    for number, state in left["optimizer"]["state"].items():
        for key, value in state.items():
            torch.testing.assert_close(value, right["optimizer"]["state"][number][key], rtol=0, atol=0)


class ModularDiagnosticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def config(self):
        return RunConfig(model="gptmini", optimizer="amsgradw", prime=7,
                         train_fraction=.5, width=8, heads=1, layers=1,
                         steps=12, eval_every=3, batch_size=8, device="cpu")

    def test_observations_do_not_change_training_or_canonical_scores(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            plain = train(self.config(), root / "plain")
            observed = train(self.config(), root / "observed", diagnostics=DiagnosticsConfig(3, True))
            left = torch.load(root / "plain/checkpoint.pt", weights_only=True)
            right = torch.load(root / "observed/checkpoint.pt", weights_only=True)
            assert_checkpoint_equal(self, left, right)
            for p, q in zip(plain["history"], observed["history"]):
                self.assertEqual(p["step"], q["step"])
                for split in ("train", "heldout"):
                    self.assertEqual(p[split], q[split])
                    # The independent float32 reductions need a rounding tolerance.
                    self.assertTrue(math.isclose(2 * q[split]["loss"],
                        q[split]["answer_loss"] + q[split]["EOS_loss"], rel_tol=1e-6, abs_tol=1e-7))
            rows = read_rows(root / "observed/diagnostics.jsonl")
            self.assertEqual({row["batch_size"] for row in rows}, {5, 8})
            self.assertTrue(all(row["temperatures"] for row in rows))
            self.assertEqual([p["step"] for p in observed["history"]], [0, 3, 6, 9, 12])
            self.assertEqual([p["step"] for p in read_rows(root / "observed/probes.jsonl")],
                             [1, 2, 4, 5, 7, 8, 10, 11])

    def test_resume_discards_uncheckpointed_logs_and_preserves_moments(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            diagnostics = DiagnosticsConfig(1, True)
            train(self.config(), root / "full", diagnostics=diagnostics)
            train(replace(self.config(), steps=6), root / "resumed", diagnostics=diagnostics)
            for name in ("diagnostics.jsonl", "probes.jsonl"):
                with (root / "resumed" / name).open("a") as stream:
                    stream.write('{"step": 7, "uncheckpointed": true}\n')
            train(self.config(), root / "resumed", resume=True, diagnostics=diagnostics)
            left = torch.load(root / "full/checkpoint.pt", weights_only=True)
            right = torch.load(root / "resumed/checkpoint.pt", weights_only=True)
            assert_checkpoint_equal(self, left, right)
            self.assertEqual(read_rows(root / "full/diagnostics.jsonl"),
                             read_rows(root / "resumed/diagnostics.jsonl"))
            probes = read_rows(root / "resumed/probes.jsonl")
            self.assertEqual([p["step"] for p in probes], [1, 2, 4, 5, 7, 8, 10, 11])
            with self.assertRaisesRegex(ValueError, "instrumentation"):
                train(self.config(), root / "resumed", resume=True)
            plan_path = root / "resumed/plan.json"
            plan = json.loads(plan_path.read_text())
            plan["source_hashes"]["corrupt"] = "different"
            plan_path.write_text(json.dumps(plan))
            with self.assertRaisesRegex(ValueError, "sources"):
                train(self.config(), root / "resumed", resume=True, diagnostics=diagnostics)

    def test_instrumented_run_cannot_be_adopted_into_plain_campaign(self):
        with tempfile.TemporaryDirectory() as root:
            config = replace(self.config(), model="reference", optimizer="adamw", steps=1)
            train(config, Path(root) / "reference-adamw-seed0", diagnostics=DiagnosticsConfig(1))
            with self.assertRaisesRegex(ValueError, "Instrumented calibration"):
                run_campaign(root, config, (0,), adopt_completed=True)

    def test_parameter_update_and_raw_moment_norms_have_independent_values(self):
        model = nn.Linear(1, 1, bias=False).double()
        model.weight.data.fill_(2)
        model.weight.grad = torch.full_like(model.weight, 3)
        optimizer = CoordinateOptimizer(model.named_parameters(), .01, decay=.2)
        output, targets = torch.tensor([[[1., 0.], [0., 1.]]]), torch.tensor([[0, 1]])
        before, losses = before_update(model, output, targets)
        optimizer.step()
        measured = after_update(model, optimizer, before, losses, torch.tensor(3.))
        norms = measured["parameters"]["weight"]
        self.assertAlmostEqual(norms["m_l2"], .3, places=12)
        self.assertAlmostEqual(norms["v_l2"], .009, places=12)
        self.assertAlmostEqual(norms["maximum_l2"], .009, places=12)
        expected_update = .004 + .01 * .3 / (.009 ** .5 + 1e-8)
        self.assertAlmostEqual(norms["update_l2"], expected_update, places=12)
        self.assertAlmostEqual(norms["parameter_l2"], 2 - expected_update, places=12)

    def test_source_fingerprints_are_relative_to_the_checkout(self):
        hashes = source_hashes()
        self.assertTrue(all(not Path(name).is_absolute() for name in hashes))
        self.assertIn("experiments/gpt_mini.py", hashes)
        self.assertIn("experiments/optimizer_benchmark/common.py", hashes)
        self.assertIn("experiments/synthetic_trainers/paper_reproduction/diagnostics.py", hashes)
