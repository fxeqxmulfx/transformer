"""Stopping contracts written before the controller and GPU training."""

import math
import unittest

from gpt_mini.infrastructure.benchmark.patience_benchmark.stopping import StopConfig, EarlyStopper, choose_best_step


class StoppingTests(unittest.TestCase):
    def config(self, **changes):
        values = dict(every=1, patience=3, min_delta=0.01, min_steps=1,
                      max_steps=20, divergence_delta=0.1, divergence_patience=2)
        return StopConfig(**(values | changes))

    def test_plateau_stops_at_patience_not_at_first_flat_check(self):
        stop = EarlyStopper(self.config())
        self.assertTrue(stop.observe(0, 2.0).new_best)
        self.assertFalse(stop.observe(1, 1.5).should_stop)
        self.assertFalse(stop.observe(2, 1.5).should_stop)
        self.assertFalse(stop.observe(3, 1.5).should_stop)
        decision = stop.observe(4, 1.5)
        self.assertTrue(decision.should_stop)
        self.assertEqual(decision.reason, "patience")
        self.assertEqual(stop.best_step, 1)

    def test_meaningful_improvement_resets_patience(self):
        stop = EarlyStopper(self.config())
        for step, loss in enumerate((2.0, 1.5, 1.5, 1.5, 1.4, 1.4, 1.4)):
            self.assertFalse(stop.observe(step, loss).should_stop)
        self.assertEqual(stop.observe(7, 1.4).reason, "patience")

    def test_tiny_improvements_save_best_but_accumulate_against_anchor(self):
        stop = EarlyStopper(self.config())
        stop.observe(0, 1.0)
        for step, loss in ((1, 0.996), (2, 0.992), (3, 0.988)):
            decision = stop.observe(step, loss)
            self.assertTrue(decision.new_best)
            self.assertFalse(decision.should_stop)
        self.assertEqual(stop.bad_checks, 0)
        self.assertEqual(stop.best_step, 3)
        stop.observe(4, 0.987)
        stop.observe(5, 0.986)
        decision = stop.observe(6, 0.985)
        self.assertTrue(decision.new_best)
        self.assertEqual(decision.reason, "patience")
        self.assertEqual(stop.best_step, 6)

    def test_sustained_divergence_uses_best_loss_and_resets_after_recovery(self):
        stop = EarlyStopper(self.config(patience=10))
        stop.observe(0, 2.0)
        stop.observe(1, 1.5)
        self.assertFalse(stop.observe(2, 1.7).should_stop)
        self.assertFalse(stop.observe(3, 1.55).should_stop)
        self.assertFalse(stop.observe(4, 1.7).should_stop)
        self.assertEqual(stop.observe(5, 1.8).reason, "validation_divergence")

    def test_minimum_budget_does_not_discard_the_actual_best(self):
        stop = EarlyStopper(self.config(min_steps=5, patience=2))
        stop.observe(0, 1.0)
        for step in range(1, 5):
            self.assertFalse(stop.observe(step, 1.05).should_stop)
        self.assertEqual(stop.observe(5, 1.05).reason, "patience")
        self.assertEqual(stop.best_step, 0)

    def test_improving_run_is_marked_capped_not_converged(self):
        stop = EarlyStopper(self.config(max_steps=4))
        for step, loss in enumerate((2.0, 1.8, 1.6, 1.4)):
            self.assertFalse(stop.observe(step, loss).should_stop)
        decision = stop.observe(4, 1.2)
        self.assertTrue(decision.new_best)
        self.assertEqual(decision.reason, "max_steps")

    def test_nonfinite_stops_immediately_and_keeps_previous_best(self):
        for invalid in (math.nan, math.inf, -math.inf):
            stop = EarlyStopper(self.config(min_steps=10))
            stop.observe(0, 1.0)
            decision = stop.observe(1, invalid)
            self.assertEqual(decision.reason, "nonfinite_validation")
            self.assertEqual((stop.best_step, stop.best_loss), (0, 1.0))

    def test_repeated_and_out_of_order_checks_and_bad_config_are_rejected(self):
        stop = EarlyStopper(self.config())
        stop.observe(1, 1.0)
        for step in (1, 0, -1):
            with self.assertRaises(ValueError):
                stop.observe(step, 0.5)
        for change in ({"patience": 0}, {"every": 0}, {"min_delta": -1},
                       {"max_steps": 0}, {"min_steps": 30},
                       {"divergence_delta": math.nan}, {"min_delta": math.inf}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.config(**change)

    def test_checkpoint_selection_ignores_test_loss_and_preserves_first_tie(self):
        curves = [dict(step=0, validation_loss=2.0, test_loss=0.1),
                  dict(step=1, validation_loss=1.5, test_loss=100),
                  dict(step=2, validation_loss=1.7, test_loss=0),
                  dict(step=3, validation_loss=1.5, test_loss=-100)]
        self.assertEqual(choose_best_step(curves), 1)
