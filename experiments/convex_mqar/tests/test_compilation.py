"""Compiled loss preserves model checkpoints, schedule, and numerical gradients."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from convex_mqar import benchmark, engine, kernels
from convex_mqar.rope import RopeTransformer
from convex_mqar.runtime import record_execution

from .support import batch, config


class CompilationTests(unittest.TestCase):
    def test_compiled_autograd_matches_eager_with_unwrapped_state_dict(self):
        torch.manual_seed(91)
        model, cfg = RopeTransformer(16, 8), config()
        data = batch(count=3, length=8, vocab=16)
        objective = kernels.loss_for(model, cfg)
        expected = objective(*data)
        expected.backward()
        gradients = {name: p.grad.clone() for name, p in model.named_parameters()}
        state_names = set(model.state_dict())
        model.zero_grad(set_to_none=True)
        compile_function = torch.compile
        with patch.object(kernels.torch, "compile", side_effect=lambda f, **kw:
                          compile_function(f, backend="aot_eager", **kw)):
            actual = kernels.loss_for(model, cfg, compiled=True)(*data)
            actual.backward()
        torch.testing.assert_close(actual, expected)
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(parameter.grad, gradients[name])
        self.assertEqual(set(model.state_dict()), state_names)
        self.assertFalse(any("_orig_mod" in name for name in model.state_dict()))

    def test_resume_switches_only_the_loss_kernel_and_records_both_segments(self):
        cfg, data = config(), batch(count=7, length=8, vocab=16)
        evaluate = engine.evaluate
        count = 0

        def interrupted(*args, **kwargs):
            nonlocal count
            count += 1
            if count == 2:
                raise InterruptedError("Resume after first complete epoch")
            return evaluate(*args, **kwargs)

        with tempfile.TemporaryDirectory() as root, redirect_stdout(io.StringIO()):
            with patch.object(engine, "evaluate", side_effect=interrupted):
                with self.assertRaises(InterruptedError):
                    engine.train_run(cfg, 8, 8, 0.003, data, data, root, Path(root) / "status.json")
            # The real compiled gradient is checked above; here a spy verifies
            # the resume transition without requiring Inductor on the test CPU.
            with patch.object(kernels.torch, "compile", side_effect=lambda f, **kw: f) as compiler:
                result, best = engine.train_run(cfg, 8, 8, 0.003, data, data, root,
                                                Path(root) / "status.json", compiled=True)
                compiler.assert_called_once()
            self.assertEqual(result["execution_segments"], [
                {"first_epoch": 1, "last_epoch": 1, "compiled_loss": False},
                {"first_epoch": 2, "last_epoch": 3, "compiled_loss": True}])
            state = torch.load(best, weights_only=True)
            model = RopeTransformer(16, 8)
            model.load_state_dict(state, strict=True)
            current = torch.load(Path(root) / "current.pt", weights_only=True)
            self.assertEqual(current["next_epoch"], 3)
            self.assertEqual(current["execution_segments"], result["execution_segments"])

    def test_legacy_environment_is_preserved_in_execution_history(self):
        with tempfile.TemporaryDirectory() as root:
            output = Path(root)
            old = {"source_sha256": "original-source", "precision": "bf16"}
            (output / "environment.json").write_text(json.dumps(old))
            environment, history = record_execution(output, config(), compiled=True)
            self.assertEqual(history[0]["environment"], old)
            self.assertTrue(history[-1]["environment"]["compiled_loss"])
            self.assertEqual(json.loads((output / "environment.json").read_text()), environment)
            self.assertIn("kernels.py", environment["fingerprinted_files"])
            self.assertIn("benchmark.py", environment["fingerprinted_files"])

    def test_completed_comparisons_are_preserved_and_not_retested_on_resume(self):
        cfg = config(epochs=1, learning_rates=(0.003,))
        with tempfile.TemporaryDirectory() as root, redirect_stdout(io.StringIO()):
            output = Path(root) / "output"
            with patch.object(benchmark.torch, "set_num_threads"):
                original = benchmark.run(cfg, output, Path(root) / "data")
            with patch.object(benchmark.torch, "set_num_threads"), \
                    patch.object(benchmark, "load_split", side_effect=AssertionError("Already compared")), \
                    patch.object(benchmark, "train_run", side_effect=AssertionError("Already trained")), \
                    patch.object(benchmark, "evaluate", side_effect=AssertionError("Do not retest")):
                restored = benchmark.run(cfg, output, Path(root) / "data", compiled=True)
            self.assertEqual(restored["results"], original["results"])
            self.assertTrue(restored["environment"]["compiled_loss"])
            self.assertEqual(len(restored["execution_history"]), 2)
