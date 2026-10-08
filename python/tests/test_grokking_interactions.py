"""Compositional controls and noninterference for pairwise head removals."""

import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from lab.domain.spec import swap
from lab.infrastructure.benchmarks.modular import ModularTask
from lab.infrastructure.engine.grokking import GrokkingObserver
from lab.infrastructure.loader import load
from lab.infrastructure.nn import build_model

STUDY = Path(__file__).resolve().parents[2] / "experiments/grokking_internals"
sys.path.insert(0, str(STUDY))
from pair_interactions import corners_contrast, measure, metrics, remove_heads
from probes.capture import evaluating, forward


class ContrastTests(unittest.TestCase):
    def test_true_product_has_negative_contrast_and_both_removals_hurt(self):
        def loss(x, y):
            return float(F.softplus(torch.tensor(-x * y, dtype=torch.float64)))
        result = corners_contrast(loss(1, 1), loss(0, 1), loss(1, 0), loss(0, 0))
        self.assertTrue(result["negative_interaction"])
        self.assertTrue(result["both_single_removals_hurt"])

    def test_negative_contrast_can_come_from_opposed_additive_scores(self):
        def loss(x, y):
            return float(F.softplus(torch.tensor(-(x + y), dtype=torch.float64)))
        result = corners_contrast(loss(1, -1), loss(0, -1), loss(1, 0), loss(0, 0))
        self.assertTrue(result["negative_interaction"])
        self.assertFalse(result["both_single_removals_hurt"])

    def test_redundant_additive_components_can_both_be_useful(self):
        def loss(x, y):
            return float(F.softplus(torch.tensor(-(x + y), dtype=torch.float64)))
        result = corners_contrast(loss(1, 1), loss(0, 1), loss(1, 0), loss(0, 0))
        self.assertGreater(result["interaction"], 0)
        self.assertTrue(result["both_single_removals_hurt"])
        self.assertFalse(result["negative_interaction"])

    def test_empty_scopes_and_zero_quotient_confounds_stay_explicit(self):
        logits = torch.tensor([[4., 0.], [0., 4.]])
        result = metrics(logits, torch.tensor([0, 0]),
                         {"all": torch.arange(2), "nonzero": torch.tensor([1]),
                          "empty": torch.tensor([], dtype=torch.long)}, logits)
        self.assertEqual(result["all"]["accuracy"], .5)
        self.assertEqual(result["nonzero"]["accuracy"], 0)
        self.assertIsNone(result["empty"]["loss"])
        with self.assertRaises(FloatingPointError):
            metrics(logits * float("nan"), torch.tensor([0, 0]), {}, logits)


class PairRemovalTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        spec = dict(load(STUDY).select([]))["gptmini-seed1"]
        for path, value in (("model.width", 16), ("model.block.attention.heads", 2),
                            ("benchmark.prime", 5)):
            spec = swap(spec, path, value)
        self.task = ModularTask(spec.benchmark, spec.seeds.data, torch.device("cpu"))
        self.model = build_model(spec.model, self.task.vocab, 1)
        self.orbit = GrokkingObserver(spec.diagnostics, self.task)

    def observe(self):
        o = self.orbit
        return measure(self.model, o.inputs, o.targets, o.train, o.heldout, o.prime, 2)

    def test_same_and_cross_block_pairs_match_zeroed_projection_columns(self):
        for addresses in ([(0, 0), (0, 1)], [(0, 0), (1, 1)]):
            other = copy.deepcopy(self.model)
            with torch.no_grad():
                for block, head in addresses:
                    other.blocks[block].attention.output.weight[:, head * 8:(head + 1) * 8] = 0
            with evaluating(self.model), evaluating(other):
                with remove_heads(self.model, addresses, 2):
                    actual = forward(self.model, self.orbit.inputs)
                expected = forward(other, self.orbit.inputs)
            torch.testing.assert_close(actual, expected)

    def test_all_pairs_are_enumerated_and_current_predictions_are_reported(self):
        result = self.observe()
        self.assertEqual(result["head_count"], 4)
        self.assertEqual(len(result["singles"]), 4)
        self.assertEqual(len(result["pairs"]), 6)
        self.assertEqual(len({(p["a"], p["b"]) for p in result["pairs"]}), 6)
        self.assertEqual(result["baseline"]["heldout_all"]["prediction_change_fraction"], 0)
        count = sum(int(index >= 4) for index in self.orbit.heldout)
        self.assertEqual(result["baseline"]["heldout_nonzero"]["examples"], count)

    def test_modes_rng_buffers_weights_and_gradients_are_preserved(self):
        self.model.train()
        self.model.blocks[0].eval()
        modes = [module.training for module in self.model.modules()]
        for parameter in self.model.parameters():
            parameter.grad = torch.ones_like(parameter)
        before = copy.deepcopy(self.model.state_dict())
        gradients = [parameter.grad.clone() for parameter in self.model.parameters()]
        rng = torch.random.get_rng_state().clone()
        self.observe()
        self.assertEqual(modes, [module.training for module in self.model.modules()])
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        for name, value in self.model.state_dict().items():
            self.assertTrue(torch.equal(before[name], value), name)
        for parameter, gradient in zip(self.model.parameters(), gradients):
            self.assertTrue(torch.equal(gradient, parameter.grad))
        self.assertFalse(any(module._forward_hooks or module._forward_pre_hooks
                             for module in self.model.modules()))

    def test_failure_restores_both_hooks_modes_and_rng(self):
        modes = [module.training for module in self.model.modules()]
        rng = torch.random.get_rng_state().clone()

        def fail(*args):
            torch.rand(1)
            raise RuntimeError("pair failure")

        with evaluating(self.model), remove_heads(self.model, [(0, 0), (1, 1)], 2):
            with patch.object(self.model.readout, "forward", fail), self.assertRaisesRegex(RuntimeError, "pair failure"):
                forward(self.model, self.orbit.inputs)
        self.assertEqual(modes, [module.training for module in self.model.modules()])
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        self.assertFalse(any(module._forward_pre_hooks for module in self.model.modules()))

    def test_invalid_addresses_are_rejected_before_any_hook_is_installed(self):
        for addresses in ([(0, 0), (0, 0)], [(0, -1)], [(2, 0)]):
            with self.assertRaises(ValueError), remove_heads(self.model, addresses, 2):
                pass
        self.assertFalse(any(module._forward_pre_hooks for module in self.model.modules()))


if __name__ == "__main__":
    unittest.main()
