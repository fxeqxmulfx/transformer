"""A continuation changes declared hyperparameters without resetting history."""

import hashlib
from pathlib import Path
import tempfile
import unittest

import torch

from lab.dsl import ScheduleFreeContinuation, describe, swap
from test_checkpoint_history import recipe, train


class ScheduleFreeContinuationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def source(self, root):
        spec = recipe()
        train(spec, root)
        path = Path(root) / "run/checkpoints/step-00000004.pt"
        return spec, path, hashlib.sha256(path.read_bytes()).hexdigest()

    def test_continuation_matches_original_and_preserves_initial_state(self):
        with tempfile.TemporaryDirectory() as source, tempfile.TemporaryDirectory() as target:
            spec, path, digest = self.source(source)
            before = path.read_bytes()
            continued = swap(spec, "checkpoint", ScheduleFreeContinuation(2, str(path), digest))
            train(continued, target)
            initial = torch.load(Path(target) / "run/checkpoints/step-00000004.pt", weights_only=True)
            pinned = torch.load(path, weights_only=True)
            actual = torch.load(Path(target) / "run/checkpoint.pt", weights_only=True)
            expected = torch.load(Path(source) / "run/checkpoint.pt", weights_only=True)
            for got, want in ((initial, pinned), (actual, expected)):
                for key, value in want["model"].items():
                    torch.testing.assert_close(got["model"][key], value, rtol=0, atol=0)
                for index, state in want["optimizer"]["state"].items():
                    for key, value in state.items():
                        torch.testing.assert_close(got["optimizer"]["state"][index][key], value, rtol=0, atol=0)
                self.assertEqual(got["cursor"], want["cursor"])
                torch.testing.assert_close(got["permutation"], want["permutation"], rtol=0, atol=0)
                self.assertEqual(got["optimizer"]["param_groups"][0]["total"],
                                 want["optimizer"]["param_groups"][0]["total"])
            self.assertEqual(before, path.read_bytes())
            self.assertEqual(initial["step"], 4)
            self.assertEqual(sorted(p.name for p in (Path(target) / "run/checkpoints").iterdir()),
                             ["step-00000004.pt", "step-00000006.pt"])

    def test_decay_override_retains_moments_and_changes_future(self):
        with tempfile.TemporaryDirectory() as source, tempfile.TemporaryDirectory() as target:
            spec, path, digest = self.source(source)
            spec = swap(spec, "checkpoint", ScheduleFreeContinuation(2, str(path), digest))
            spec = swap(spec, "optimizer.weight_decay", 0.)
            train(spec, target)
            pinned = torch.load(path, weights_only=True)
            initial = torch.load(Path(target) / "run/checkpoints/step-00000004.pt", weights_only=True)
            final = torch.load(Path(target) / "run/checkpoint.pt", weights_only=True)
            original = torch.load(Path(source) / "run/checkpoint.pt", weights_only=True)
            self.assertEqual(initial["optimizer"]["param_groups"][0]["weight_decay"], 0.)
            self.assertEqual(initial["optimizer"]["param_groups"][0]["step"], 4)
            for index, state in pinned["optimizer"]["state"].items():
                for key, value in state.items():
                    torch.testing.assert_close(initial["optimizer"]["state"][index][key], value, rtol=0, atol=0)
            self.assertTrue(any(not torch.equal(value, original["model"][key])
                                for key, value in final["model"].items()))

    def test_wrong_digest_or_moment_parameters_are_rejected(self):
        with tempfile.TemporaryDirectory() as source:
            spec, path, digest = self.source(source)
            for digest_value, betas, message in (("0"*64, spec.optimizer.betas, "SHA256"),
                                                (digest, (.8, .98), "betas")):
                with tempfile.TemporaryDirectory() as target:
                    variant = swap(spec, "checkpoint", ScheduleFreeContinuation(2, str(path), digest_value))
                    variant = swap(variant, "optimizer.betas", betas)
                    with self.assertRaisesRegex(ValueError, message):
                        train(variant, target)

    def test_policy_keeps_old_checkpoint_descriptions_unchanged(self):
        self.assertEqual(describe(recipe().checkpoint), {"type": "CheckpointHistory", "every": 2})
        with self.assertRaises(ValueError):
            ScheduleFreeContinuation()


if __name__ == "__main__":
    unittest.main()
