"""Specified diversity coverage, actual parameter counts, and train-only N choice."""

import itertools
import math
import unittest

from experiments.gpt_mini import GPTMini
from experiments.synthetic_trainers.config import ModelSpec
from experiments.synthetic_trainers.data import build_split
from experiments.synthetic_trainers.scaling_calculations import critical_sample_choice, motif_bound, motif_coverage, parameter_count
from experiments.synthetic_trainers.specs import TaskSpec


class ScalingCalculationsTests(unittest.TestCase):
    def test_required_even_pool_meets_the_probability_bound_and_smaller_one_does_not(self):
        bound = motif_bound()
        def failure(rows):
            return bound["events"] * (1 - bound["minimum_event_probability"]) ** (rows // 2)
        self.assertLessEqual(failure(bound["required_rows"]), .05)
        self.assertGreater(failure(bound["required_rows"] - 2), .05)
        self.assertLess(failure(bound["rounded_rows"]), .001)
        self.assertEqual(bound["events"], 6960)

    def test_parameter_formula_matches_real_tied_models(self):
        for width, layers in itertools.product((16, 64), (2, 6)):
            model = ModelSpec(width=width, layers=layers, heads=8).reference_config(38, 67)
            actual = sum(parameter.numel() for parameter in GPTMini(model).parameters())
            self.assertEqual(parameter_count(38, width, layers, 8), actual)

    def test_actual_coverage_matches_an_independent_enumeration(self):
        split = build_split(TaskSpec(task="copy", length=4, min_length=1, symbols=2), "train", 1, 1000)
        reported = motif_coverage(split, 2)
        expected = {(length, position, tuple(word)) for length in range(2, 5)
                    for position in range(length - 1) for word in itertools.product((36, 37), repeat=2)}
        seen = {(len(row.prompt) - 2, position, row.prompt[position + 1:position + 3])
                for row in split.examples for position in range(len(row.prompt) - 3)}
        self.assertEqual(reported["events"], len(expected))
        self.assertEqual(reported["observed_events"], len(seen & expected))
        self.assertEqual(reported["missing_events"], 0)

    def test_calibration_uses_train_fit_and_preserves_nonmonotone_holes(self):
        rows = [{"train_examples": n, "final_fitted": fit, "final_loss": loss} for n, fit, loss in
                ((64, True, 9), (256, False, 0), (1024, True, 7), (4096, False, 1))]
        choice = critical_sample_choice(rows)
        self.assertEqual(choice["chosen_sample_size"], 2048)
        self.assertEqual(choice["largest_observed_fitted_pool"], 1024)
        self.assertTrue(choice["nonmonotone_fit"])
        for row in rows:
            row["final_loss"] = -row["final_loss"]
        self.assertEqual(choice, critical_sample_choice(rows))
