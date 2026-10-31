"""Check actual state replay and all-parameter directional derivatives."""

from pathlib import Path
import runpy
import tempfile
import unittest

import torch

from lab.infrastructure.benchmarks import build_task
from lab.infrastructure.optim import build_optimizer
from test_checkpoint_history import recipe, train

ANALYSIS = runpy.run_path(str(Path(__file__).resolve().parents[2] /
                            "experiments/rotate90_transfer/analyze.py"))


class LearningTransferTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_replay_restores_sampler_and_both_optimizer_time_regimes(self):
        with tempfile.TemporaryDirectory() as root:
            spec = recipe()
            train(spec, root)
            folder = Path(root) / "run" / "checkpoints"
            for step in (0, 6):
                path = folder / f"step-{step:08d}.pt"
                before = path.read_bytes()
                result = ANALYSIS["probe"](spec, path)
                self.assertEqual(result["step"], step)
                self.assertEqual(result["next_update"], step + 1)
                self.assertLess(result["evaluation_identity_max_error"], 1e-12)
                terms = result["directional_terms"]
                self.assertAlmostEqual(terms["delta"], sum(terms[k] for k in ("averaging", "transfer", "decay")), places=12)
                self.assertEqual(result, ANALYSIS["probe"](spec, path))
                self.assertEqual(before, path.read_bytes())

    def test_joint_derivative_matches_finite_difference_with_all_tied_occurrences(self):
        spec = recipe()
        model = ANALYSIS["double_model"](spec)
        optimizer = build_optimizer(spec.optimizer, model)
        task = build_task(spec.benchmark, spec.seeds.data, torch.device("cpu"))
        ANALYSIS["measure"](model, task.splits["train"][:8], gradient=True)
        pieces, _ = ANALYSIS["displacement"](optimizer, .003)
        ANALYSIS["measure"](model, task.splits["heldout"], gradient=True)
        derivative = sum(float((p.grad * piece["delta"]).sum())
                         for p, piece in zip(model.parameters(), pieces, strict=True))
        self.assertTrue(all(p.grad.norm() > 0 for p in model.parameters()))
        losses = []
        h = 1e-3
        for sign in (-1, 1):
            with torch.no_grad():
                for p, piece in zip(model.parameters(), pieces, strict=True):
                    p.copy_(piece["x"] + sign * h * piece["delta"])
            losses.append(ANALYSIS["measure"](model, task.splits["heldout"])["loss"])
        self.assertAlmostEqual(derivative, (losses[1] - losses[0]) / (2 * h), delta=1e-6)


if __name__ == "__main__":
    unittest.main()
