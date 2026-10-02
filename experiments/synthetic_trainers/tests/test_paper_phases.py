"""Transient train fits and late collapses cannot become a stable grokking claim."""

import unittest

from experiments.synthetic_trainers.paper_phases import diagnose


def report(points):
    return {"plan": {"config": {"target": .99, "patience": 2}},
            "history": [{"step": step, "train": {"accuracy": train}, "heldout": {"accuracy": heldout}}
                        for step, train, heldout in points]}


class PaperPhaseTests(unittest.TestCase):
    def test_error_double_descent_can_coincide_with_delayed_generalization(self):
        measured = report(((0, 0, 0), (1, .5, .6), (2, 1, .01),
                           (3, 1, .02), (4, 1, 1), (5, 1, 1), (6, 1, 1)))
        phase = diagnose(measured)
        curve = phase["epoch_error_curve_before_generalization"]
        self.assertEqual([curve[name]["step"] for name in ("initial", "first_minimum", "peak", "final")], [0, 1, 2, 6])
        self.assertAlmostEqual(curve["first_descent"], .6)
        self.assertAlmostEqual(curve["peak_rise"], .59)
        self.assertAlmostEqual(curve["second_descent"], .99)
        self.assertTrue(phase["both_epoch_error_double_descent_and_grokking"])

    def test_late_instability_cannot_become_an_overfitting_peak_before_generalization(self):
        measured = report(((0, 0, 0), (1, .5, .6), (2, 1, .6), (3, 1, .6),
                           (4, 1, 1), (5, 1, 1), (6, 1, .01), (7, 1, 1)))
        phase = diagnose(measured)
        curve = phase["epoch_error_curve_before_generalization"]
        self.assertLess(curve["peak"]["step"], phase["sustained_heldout_target"]["onset"])
        self.assertFalse(curve["full_error_double_descent"])
        self.assertFalse(phase["both_epoch_error_double_descent_and_grokking"])

    def test_error_recovery_without_the_generalization_target_is_not_grokking(self):
        measured = report(((0, 0, 0), (1, .5, .6), (2, 1, .01),
                           (3, 1, .01), (4, 1, .1), (5, 1, .1)))
        phase = diagnose(measured)
        self.assertTrue(phase["epoch_error_curve_before_generalization"]["full_error_double_descent"])
        self.assertFalse(phase["both_epoch_error_double_descent_and_grokking"])

    def test_target_cost_uses_observed_onset_and_confirmation_and_keeps_failures_missing(self):
        measured = report(((0, 0, 0), (10, 1, .01), (30, 1, .02),
                           (50, 1, 1), (80, 1, .9), (100, 1, 1), (140, 1, 1)))
        for point, training, wall in zip(measured["history"], (0, 2, 9, 14, 20, 28, 40), (1, 4, 13, 19, 27, 37, 53)):
            point.update(training_seconds=training, wall_seconds=wall)
        phase = diagnose(measured)
        self.assertEqual(phase["time_to_sustained_train_fit"]["onset"],
                         {"training_seconds": 2, "wall_seconds": 4})
        self.assertEqual(phase["time_to_sustained_heldout_target"],
                         {"onset": {"training_seconds": 28, "wall_seconds": 37},
                          "confirmed": {"training_seconds": 40, "wall_seconds": 53}})
        measured["history"][-1]["heldout"]["accuracy"] = .5
        self.assertIsNone(diagnose(measured)["time_to_sustained_heldout_target"])

    def test_transient_first_fit_does_not_create_a_memorization_plateau(self):
        measured = report(((0, 0, 0), (1, 1, .01), (2, .5, .01), (3, 1, .9), (4, 1, 1), (5, 1, 1)))
        phase = diagnose(measured)
        self.assertEqual(phase["sustained_train_fit"], {"onset": 3, "confirmed": 4})
        self.assertEqual(phase["lag_after_sustained_train_fit"], 1)
        self.assertIsNone(phase["memorization_plateau"])
        self.assertFalse(phase["observed_plateau_then_generalization"])

    def test_sustained_plateau_and_late_failure_are_reported_separately(self):
        measured = report(((0, 0, 0), (1, 1, .01), (2, 1, .02), (3, 1, .8), (4, 1, 1), (5, 1, 1), (6, 1, 1)))
        phase = diagnose(measured)
        self.assertTrue(phase["observed_plateau_then_generalization"])
        self.assertEqual(phase["memorization_plateau"]["observations"], 2)
        self.assertEqual(phase["lag_after_sustained_train_fit"], 3)
        self.assertEqual(phase["fraction_observations_at_target_after_confirmation"], 1)
        measured["history"][-1]["heldout"]["accuracy"] = .1
        phase = diagnose(measured)
        self.assertFalse(phase["observed_plateau_then_generalization"])
        self.assertEqual(phase["minimum_heldout_accuracy_after_confirmation"], .1)
        self.assertEqual(phase["fraction_observations_at_target_after_confirmation"], .5)
