"""Check the response formula against real source updates, including raw moments."""

from pathlib import Path
import runpy
import tempfile
import unittest

import torch

from test_checkpoint_history import recipe, train

DYNAMICS = runpy.run_path(str(Path(__file__).resolve().parents[2] /
                            "experiments/rotate90_forcing/dynamics.py"))


class GradientForcingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_forced_and_zero_gradient_paths_match_actual_source(self):
        with tempfile.TemporaryDirectory() as root:
            spec = recipe()
            train(spec, root)
            path = Path(root) / "run/checkpoints/step-00000006.pt"
            before = path.read_bytes()
            result = DYNAMICS["check_source"](spec, path, 32)
            for key in ("forced_buffer_max_error", "homogeneous_buffer_max_error", "evaluation_max_error"):
                self.assertLess(result[key], 1e-12)
            self.assertLess(result["zero_gradient_plane"]["relative_residual"], 1e-12)
            self.assertGreater(result["zero_gradient_second_change"], 0.)
            self.assertGreater(result["response_evaluation_norm"], .01)
            self.assertEqual(before, path.read_bytes())

    def test_plane_distance_detects_new_parameter_direction(self):
        y, z = [torch.tensor([1., 0., 0.])], [torch.tensor([0., 1., 0.])]
        inside = DYNAMICS["plane_residual"]([torch.tensor([2., -3., 0.])], y, z)
        outside = DYNAMICS["plane_residual"]([torch.tensor([0., 0., 1.])], y, z)
        self.assertEqual(inside["basis_rank"], 2)
        self.assertEqual(inside["relative_residual"], 0.)
        self.assertEqual(outside["relative_residual"], 1.)


if __name__ == "__main__":
    unittest.main()
