"""Before-loss controls and stage-capture noninterference on real blocks."""

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
from output_interactions import measure, node_statistics, observe, reconstruction_metrics
from pair_interactions import corners_contrast


class OutputContrastTests(unittest.TestCase):
    def scopes(self, count):
        return {"all": torch.arange(count)}

    def corners(self):
        return [torch.tensor(value, dtype=torch.float64) for value in
                ([[1., 0.], [1., 0.]], [[-1., 0.], [2., 0.]],
                 [[2., 0.], [-1., 0.]], [[0., 0.], [0., 0.]])]

    def test_additive_scores_can_have_useful_negative_CE_and_zero_output_interaction(self):
        corners = self.corners()
        targets = torch.zeros(2, dtype=torch.long)
        ce = corners_contrast(*(float(F.cross_entropy(x, targets)) for x in corners))
        self.assertTrue(ce["negative_interaction"])
        self.assertTrue(ce["both_single_removals_hurt"])
        result = node_statistics(corners, self.scopes(2), logits=True)
        self.assertEqual(result["scopes"]["all"]["interaction_mean_square"], 0)
        reconstruction = reconstruction_metrics(corners, targets, self.scopes(2))
        self.assertEqual(reconstruction["all"]["prediction_change_fraction"], 0)

    def test_independent_common_offsets_at_all_corners_are_removed(self):
        corners = self.corners()
        offsets = [torch.tensor([[value], [-2 * value]]) for value in (10., 1., 7., -3.)]
        original = node_statistics(corners, self.scopes(2), logits=True)["scopes"]["all"]
        shifted = node_statistics([x + b for x, b in zip(corners, offsets)],
                                  self.scopes(2), logits=True)["scopes"]["all"]
        for key in ("interaction_mean_square", "single_a_change_mean_square", "single_b_change_mean_square"):
            self.assertEqual(original[key], shifted[key])
        self.assertGreater(shifted["common_row_offset_rms"], 0)
        raw = node_statistics([x + b for x, b in zip(corners, offsets)], self.scopes(2))["scopes"]["all"]
        self.assertGreater(raw["interaction_mean_square"], 0)

    def test_binary_product_matches_the_Lean_sum_energy_after_coordinate_normalization(self):
        corners = [torch.tensor([[1., 0.]])] + [torch.zeros(1, 2)] * 3
        result = node_statistics(corners, self.scopes(1), logits=True)["scopes"]["all"]
        self.assertEqual(2 * result["interaction_mean_square"], .5)
        self.assertEqual(result["interaction_over_singles_energy"], .5)

    def test_positive_common_scaling_preserves_relative_interaction_energy(self):
        corners = [torch.tensor([[4., 0.]])] + [torch.tensor([[3., 0.]])] * 2 + [torch.zeros(1, 2)]
        original = node_statistics(corners, self.scopes(1), logits=True)["scopes"]["all"]
        scaled = node_statistics([4 * x for x in corners], self.scopes(1), logits=True)["scopes"]["all"]
        self.assertEqual(scaled["interaction_mean_square"], 16 * original["interaction_mean_square"])
        self.assertEqual(scaled["interaction_over_singles_energy"], original["interaction_over_singles_energy"])

    def test_nonzero_logit_interaction_can_preserve_all_decisions(self):
        corners = [torch.tensor([[v, 0.], [0., v]]) for v in (4., 3., 3., 0.)]
        result = node_statistics(corners, self.scopes(2), logits=True)["scopes"]["all"]
        self.assertGreater(result["interaction_mean_square"], 0)
        reconstruction = reconstruction_metrics(corners, torch.tensor([0, 1]), self.scopes(2))["all"]
        self.assertEqual(reconstruction["accuracy"], 1)
        self.assertEqual(reconstruction["prediction_change_fraction"], 0)
        self.assertEqual(reconstruction["target_margin_interaction_mean"], -2)

    def test_removal_endpoints_can_miss_interior_gate_interaction(self):
        def corners(a, b):
            return [torch.tensor([[u * v * (1 - u) * (1 - v), 0.]])
                    for u, v in ((a, b), (0, b), (a, 0), (0, 0))]
        endpoint = node_statistics(corners(1, 1), self.scopes(1), logits=True)["scopes"]["all"]
        interior = node_statistics(corners(.5, .5), self.scopes(1), logits=True)["scopes"]["all"]
        self.assertEqual(endpoint["interaction_mean_square"], 0)
        self.assertIsNone(endpoint["interaction_over_singles_energy"])
        self.assertEqual(2 * interior["interaction_mean_square"], 1 / 512)

    def test_empty_scopes_nonfinite_values_and_overflow_do_not_supply_evidence(self):
        scopes = {"empty": torch.tensor([], dtype=torch.long)}
        result = node_statistics(self.corners(), scopes, logits=True)["scopes"]["empty"]
        self.assertIsNone(result["interaction_mean_square"])
        self.assertIsNone(result["interaction_over_singles_energy"])
        with self.assertRaises(FloatingPointError):
            node_statistics([x * float("nan") for x in self.corners()], self.scopes(2))
        with self.assertRaises(FloatingPointError):
            node_statistics([torch.full((1, 2), 1e200, dtype=torch.float64)] * 4, self.scopes(1))


class StageCaptureTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        spec = dict(load(STUDY).select([]))["gptmini-seed1"]
        for path, value in (("model.width", 16), ("model.block.attention.heads", 2), ("benchmark.prime", 5)):
            spec = swap(spec, path, value)
        self.task = ModularTask(spec.benchmark, spec.seeds.data, torch.device("cpu"))
        self.model = build_model(spec.model, self.task.vocab, 1).double()
        self.orbit = GrokkingObserver(spec.diagnostics, self.task)

    def measure(self):
        o = self.orbit
        return measure(self.model, o.inputs, o.targets, o.train, o.heldout, o.prime, 2)

    def test_linear_same_block_projection_and_causal_earlier_stages_are_controls(self):
        result = self.measure()
        self.assertEqual(len(result["pairs"]), 6)
        pair = next(p for p in result["pairs"] if p["a"] == "blocks.0.head0" and p["b"] == "blocks.0.head1")
        early = pair["nodes"]["blocks.0.attention.output"]["scopes"]["heldout_all"]
        nonlinear = pair["nodes"]["blocks.0.ffn.output"]["scopes"]["heldout_all"]
        self.assertLess(early["interaction_mean_square"], 1e-26)
        self.assertGreater(nonlinear["interaction_mean_square"], 1e-12)
        cross = next(p for p in result["pairs"] if p["a"] == "blocks.0.head0" and p["b"] == "blocks.1.head1")
        self.assertLess(cross["nodes"]["blocks.0.attention.output"]["scopes"]["heldout_all"]["interaction_mean_square"], 1e-26)

    def test_complete_measurement_preserves_modes_rng_buffers_weights_and_gradients(self):
        self.model.train()
        self.model.blocks[0].eval()
        modes = [module.training for module in self.model.modules()]
        for parameter in self.model.parameters():
            parameter.grad = torch.ones_like(parameter)
        before = copy.deepcopy(self.model.state_dict())
        gradients = [parameter.grad.clone() for parameter in self.model.parameters()]
        rng = torch.random.get_rng_state().clone()
        self.measure()
        self.assertEqual(modes, [module.training for module in self.model.modules()])
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        for name, value in self.model.state_dict().items():
            self.assertTrue(torch.equal(before[name], value), name)
        for parameter, gradient in zip(self.model.parameters(), gradients):
            self.assertTrue(torch.equal(gradient, parameter.grad))
        self.assertFalse(any(module._forward_hooks or module._forward_pre_hooks for module in self.model.modules()))

    def test_failed_stage_capture_removes_all_hooks_and_restores_modes_rng(self):
        modes = [module.training for module in self.model.modules()]
        rng = torch.random.get_rng_state().clone()

        def fail(*args):
            torch.rand(1)
            raise RuntimeError("stage failure")

        with patch.object(self.model.readout, "forward", fail), self.assertRaisesRegex(RuntimeError, "stage failure"):
            observe(self.model, self.orbit.inputs)
        self.assertEqual(modes, [module.training for module in self.model.modules()])
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        self.assertFalse(any(module._forward_hooks or module._forward_pre_hooks for module in self.model.modules()))


if __name__ == "__main__":
    unittest.main()
