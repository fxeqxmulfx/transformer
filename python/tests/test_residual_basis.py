"""Original Basis data, masked objectives, first-crossing isolation and replay."""

import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from lab.dsl import (AdaptiveResidual, CanonicalResidual, Eager, ResidualBasis,
                     ResidualMaxAutotune, ScheduleFreeAdamW, residual_basis, swap)
from lab.domain.basis import BASIS
from lab.infrastructure.benchmarks import build_task
from lab.infrastructure.benchmarks.synthetic.training import SyntheticTask
from lab.infrastructure.benchmarks.synthetic.vocabulary import IGNORE
from lab.infrastructure.engine.residual import ResidualEagerStepper
from lab.infrastructure.nn import build_model
from test_residual_transformer import train


def tiny(task="depth", kind=AdaptiveResidual):
    spec = residual_basis(kind(8, 1, 2, 129), "easy")[task]
    for path, value in (("benchmark.train", 17), ("benchmark.validation", 8),
                        ("benchmark.test", 8), ("budget.updates", 6), ("budget.batch", 4),
                        ("evaluate.every", 2), ("evaluate.batch", 8),
                        ("checkpoint.every", 2), ("diagnostics.every", 0),
                        ("execution", Eager(device="cpu"))):
        spec = swap(spec, path, value)
    if task == "depth":
        spec = swap(spec, "benchmark.ood", ())
        spec = swap(spec, "benchmark.min_length", 8)
        spec = swap(spec, "benchmark.length", 16)
    return spec


class ResidualBasisTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_all_recipes_preserve_basis_semantics_and_observation_cadence(self):
        for mode in ("easy", "hard"):
            for name, run in residual_basis(CanonicalResidual(64, 2, 4, 129), mode).items():
                original, recipe = BASIS[mode][name]
                self.assertEqual(vars(run.benchmark), vars(original))
                self.assertEqual(run.optimizer, ScheduleFreeAdamW(
                    lr=recipe.lr, betas=(.9, .98), weight_decay=.1))
                self.assertEqual(run.budget.batch, recipe.batch)
                self.assertEqual(run.evaluate.every * recipe.batch, 3200)
                self.assertEqual(run.seeds.data, 1)
                self.assertEqual(run.seeds.batch_seed, 10000)
                self.assertIsNone(run.benchmark.selection)
                self.assertEqual(run.benchmark.final, ())
                self.assertEqual(run.benchmark.last, ())
                self.assertIsNone(run.stopping)
                self.assertEqual(run.benchmark.validation_name, original.selection)

    def test_original_splits_tokens_labels_and_fingerprints_are_unchanged(self):
        for mode in ("easy", "hard"):
            for name, (original, _) in BASIS[mode].items():
                # Small prefixes exercise every generator, including hard OOD.
                for path, value in (("train", 31), ("validation", 11), ("test", 13)):
                    original = swap(original, path, value)
                expected = build_task(original, 1, torch.device("cpu"))
                actual = build_task(ResidualBasis(**vars(original)), 1, torch.device("cpu"))
                self.assertEqual(actual.fingerprints, expected.fingerprints, (mode, name))
                for split, rows in actual.splits.items():
                    self.assertEqual(rows.examples, expected.splits[split].examples)
                    torch.testing.assert_close(rows.rows, expected.splits[split].rows, rtol=0, atol=0)

    def test_actual_gpt_masked_loss_and_every_gradient_match_full_objective(self):
        for name in ("depth", "recall", "parity"):
            spec = tiny(name)
            task = build_task(spec.benchmark, 1, torch.device("cpu"))
            model = build_model(spec.model, task.vocab, 0).double()
            reference = copy.deepcopy(model)
            batch = task.inputs(torch.arange(4), static=True)
            output, targets = task.forward(model, batch, supervised=True)
            loss = task.loss(output, targets)
            expected = F.cross_entropy(reference(batch[:, 0]).flatten(0, 1),
                                       batch[:, 1].flatten(), ignore_index=IGNORE)
            torch.testing.assert_close(loss, expected, rtol=1e-13, atol=1e-13)
            loss.backward()
            expected.backward()
            for a, b in zip(model.parameters(), reference.parameters(), strict=True):
                torch.testing.assert_close(a.grad, b.grad, rtol=1e-11, atol=1e-13)

    def test_parity_evaluation_inherits_original_free_generation_not_teacher_forcing(self):
        self.assertIs(SyntheticTask.accumulate, type(build_task(
            tiny("parity").benchmark, 1, torch.device("cpu"))).accumulate)
        spec = tiny("parity")
        task = build_task(spec.benchmark, 1, torch.device("cpu"))
        self.assertTrue(task.splits["validation"].generates)
        self.assertFalse(task.splits["train"].generates)
        # A token score of 1 does not replace exact sequence accuracy.
        self.assertFalse(spec.benchmark.stable([
            {"validation": {"sequence_accuracy": .98, "token_accuracy": 1.}}]))
        self.assertTrue(spec.benchmark.stable([
            {"validation": {"sequence_accuracy": .99, "token_accuracy": 1.}}]))

    def test_first_crossing_and_resume_never_open_test_or_choose_best(self):
        calls = []

        def score(self, name):
            calls.append(name)
            if name != "validation":
                raise AssertionError("Final test opened during learning")
            return {"sequence_accuracy": .98 if len(calls) == 1 else .99, "loss": .1}

        with tempfile.TemporaryDirectory() as directory, patch.object(ResidualEagerStepper, "evaluate", score):
            spec = tiny()
            result, run = train(spec, Path(directory))
            self.assertEqual(result["stop"], {"step": 2, "reason": "validation99"})
            saved = copy.deepcopy(run.checkpoint("cpu"))
            again, run = train(spec, Path(directory))
            self.assertEqual(again["stop"], result["stop"])
            torch.testing.assert_close(run.checkpoint("cpu")["model"], saved["model"], rtol=0, atol=0)
            self.assertEqual(calls, ["validation", "validation"])
            self.assertNotIn("best", result)

    def test_compiled_full_state_resume_preserves_epoch_remainders_and_gradients(self):
        spec = swap(tiny(), "execution", ResidualMaxAutotune(device="cpu"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            _, straight = train(spec, path / "straight")
            train(swap(spec, "budget.updates", 2), path / "continued")
            result, continued = train(spec, path / "continued")
            a, b = straight.checkpoint("cpu"), continued.checkpoint("cpu")
            for key in ("model", "optimizer", "permutation"):
                torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
            for key in ("cursor", "step", "examples_seen", "last_batch_size"):
                self.assertEqual(a[key], b[key])
            self.assertEqual(result["stop"], {"step": 6, "reason": "budget"})
            self.assertEqual(len(continued.records("gradients")), 6)
            self.assertTrue(all("test" not in row for row in continued.records("history")))


if __name__ == "__main__":
    unittest.main()
