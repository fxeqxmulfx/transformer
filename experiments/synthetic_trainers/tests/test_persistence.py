"""A final rebound or an incomplete trajectory cannot satisfy persistence."""

from copy import deepcopy
import unittest

from experiments.synthetic_trainers.persistence import PersistenceConfig, assess


def measured(heldout, train=None):
    train = [0, *([1] * (len(heldout) - 1))] if train is None else train
    history = [{"step": step * 250, "train": {"accuracy": fit},
                "heldout": {"accuracy": score}, "training_seconds": step,
                "wall_seconds": step * 2}
               for step, (fit, score) in enumerate(zip(train, heldout))]
    return {"plan": {"status": "complete", "config": {"steps": history[-1]["step"],
            "eval_every": 250, "target": .99, "patience": 2}},
            "completed_steps": history[-1]["step"], "history": history, "final": history[-1]}


class PersistenceTests(unittest.TestCase):
    def criterion(self):
        return PersistenceConfig(confirmation_observations=3, tail_steps=1000)

    def test_long_memorization_then_uninterrupted_tail_meets_both_criteria(self):
        report = measured([0, *([.01] * 5), *([1] * 7)])
        result = assess(report, self.criterion())
        self.assertTrue(result["stable_grokking"])
        self.assertTrue(result["persistent_final_performance"])
        self.assertEqual(result["plateau"], {"start": 250, "end": 1250, "observations": 5})
        self.assertEqual(result["long_confirmation"]["confirmed"], 2000)
        self.assertEqual(result["tail_observations"], 5)

    def test_rapid_generalization_is_persistent_without_a_grokking_plateau(self):
        result = assess(measured([0, *([1] * 12)]), self.criterion())
        self.assertTrue(result["persistent_final_performance"])
        self.assertFalse(result["stable_grokking"])
        self.assertIsNone(result["plateau"])

    def test_late_collapse_with_final_rebound_fails_prospective_tail(self):
        report = measured([0, *([.01] * 5), *([1] * 6), .1, 1])
        result = assess(report, self.criterion())
        self.assertIsNotNone(result["long_confirmation"])
        self.assertTrue(result["legacy_phase_diagnostics"]["final_heldout_target"])
        self.assertFalse(result["persistent_final_performance"])
        self.assertFalse(result["stable_grokking"])
        self.assertEqual(result["tail_failures"], [3000])

    def test_train_collapse_also_invalidates_joint_final_performance(self):
        report = measured([0, *([.01] * 5), *([1] * 7)])
        report["history"][-2]["train"]["accuracy"] = .9
        result = assess(report, self.criterion())
        self.assertFalse(result["persistent_final_performance"])
        self.assertEqual(result["tail_minimum_heldout_accuracy"], 1)

    def test_omitted_duplicate_unfinished_and_disagreeing_final_histories_fail(self):
        original = measured([0, *([.01] * 5), *([1] * 7)])
        for kind in ("omitted", "duplicate", "unfinished", "final"):
            with self.subTest(kind=kind):
                report = deepcopy(original)
                if kind == "omitted":
                    del report["history"][7]
                elif kind == "duplicate":
                    report["history"].insert(7, report["history"][7])
                elif kind == "unfinished":
                    report["completed_steps"] -= 1
                else:
                    report["final"] = {**report["final"], "step": 0}
                result = assess(report, self.criterion())
                self.assertFalse(result["complete_canonical_history"])
                self.assertFalse(result["persistent_final_performance"])

    def test_single_transient_low_score_and_short_budget_are_insufficient(self):
        result = assess(measured([0, .01, *([1] * 10)]), self.criterion())
        self.assertIsNone(result["plateau"])
        self.assertFalse(result["stable_grokking"])
        result = assess(measured([0, 1, 1]), self.criterion())
        self.assertFalse(result["persistent_final_performance"])

    def test_invalid_criterion_is_rejected(self):
        for fields in ({"target": 0}, {"heldout_ceiling": .99},
                       {"tail_steps": 0}, {"confirmation_observations": 0}):
            with self.assertRaises(ValueError):
                PersistenceConfig(**fields)
