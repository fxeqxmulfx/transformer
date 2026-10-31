"""Source-state replay and the full finite-stack sign symmetry."""

from pathlib import Path
import runpy
import tempfile
import unittest

import torch

from lab.dsl import swap
from test_checkpoint_history import recipe, train

MEMORY = runpy.run_path(str(Path(__file__).resolve().parents[2] /
                          "experiments/rotate90_transfer/memory.py"))


class OptimizerMemoryTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_closed_form_matches_real_gradients_and_keeps_base_fixed(self):
        with tempfile.TemporaryDirectory() as root:
            spec = recipe()
            train(spec, root)
            path = Path(root) / "run/checkpoints/step-00000006.pt"
            before = path.read_bytes()
            for count in (1, 7, 32):
                result = MEMORY["check_source"](spec, path, count)
                self.assertLess(result["max_coordinate_error"], 1e-12)
                self.assertEqual(result["base_max_change"], 0.)
                self.assertGreater(result["second_moment_change_norm"], 0.)
                self.assertGreater(result["evaluation_movement_norm"], 0.)
                self.assertLess(result["total_error"], 1e-16)
            self.assertEqual(before, path.read_bytes())

    def test_negating_all_parameters_preserves_actual_depth_three_scores(self):
        with tempfile.TemporaryDirectory() as root:
            spec = swap(recipe(), "model.depth", 3)
            train(spec, root)
            path = Path(root) / "run/checkpoints/step-00000006.pt"
            model, _, _ = MEMORY["restore"](spec, path)
            inputs = torch.tensor([[0, 1], [3, 2], [6, 5]])
            with torch.no_grad():
                logits = model(inputs).clone()
                self.assertGreater(float(logits.abs().max()), .01)
                for p in model.parameters():
                    p.neg_()
                torch.testing.assert_close(model(inputs), logits, rtol=0, atol=1e-12)
                for p in model.parameters():
                    p.zero_()
                torch.testing.assert_close(model(inputs), torch.zeros_like(logits), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
