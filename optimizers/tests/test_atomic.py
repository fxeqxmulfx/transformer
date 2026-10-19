"""Standalone atomic solver boundary cases."""

from __future__ import annotations

import unittest

import torch

from lean_optim.atomic import simplex_project, simplex_squared_fit


class AtomicTests(unittest.TestCase):
    """Numerical checks for atomic."""

    def test_simplex_projection_and_singular_face_correction(self) -> None:
        """Check simplex projection and singular face correction."""
        torch.testing.assert_close(
            simplex_project(torch.tensor([2.0, -1.0, 0.0])), torch.tensor([1.0, 0.0, 0.0])
        )
        columns = torch.tensor([[0.0, 0.0], [1.0, 2.0], [1.0, 2.0]], dtype=torch.float64)
        mass = simplex_squared_fit(columns, torch.tensor([0.25, 0.5], dtype=torch.float64))
        self.assertTrue(bool((mass >= 0).all()))
        self.assertAlmostEqual(float(mass.sum()), 1.0)
        torch.testing.assert_close(mass @ columns, torch.tensor([0.25, 0.5], dtype=torch.float64))


if __name__ == "__main__":
    unittest.main()
