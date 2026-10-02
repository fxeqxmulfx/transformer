"""Count missed targets separately from episodes and incomplete recovery windows."""

from copy import deepcopy
import unittest

from experiments.synthetic_trainers.persistence import PersistenceConfig
from experiments.synthetic_trainers.stability_recovery import describe


def trajectory(scores, train=None, through=None):
    train = [0, *([1] * (len(scores) - 1))] if train is None else train
    history = [{"step": i * 100, "train": {"accuracy": a}, "heldout": {"accuracy": b}}
               for i, (a, b) in enumerate(zip(train, scores))]
    budget = history[-1]["step"]
    if through is not None:
        history = [p for p in history if p["step"] <= through]
    return {"plan": {"status": "complete" if through is None else "running",
                     "config": {"steps": budget, "eval_every": 100, "target": .99, "patience": 2}},
            "completed_steps": history[-1]["step"], "history": history, "final": history[-1]}


class RecoveryTests(unittest.TestCase):
    def criterion(self):
        return PersistenceConfig(confirmation_observations=2, tail_steps=600)

    def test_episode_frequency_depth_recovery_and_last_inclusive_window(self):
        report = trajectory([0, .995, .995, .98, .97, .995, .995, .96, .96, .995, .995])
        result = describe(report, self.criterion(), window_steps=200)
        joint = result["metrics"]["joint"]
        self.assertFalse(result["frozen_persistent_final_performance"])
        self.assertEqual(joint["episode_count"], 2)
        self.assertEqual(joint["post_long_onset_failed_observations"], 4)
        self.assertEqual([e["failed_observations"] for e in joint["episodes_after_long_onset"]], [2, 2])
        self.assertEqual([e["first_recovery"] for e in joint["episodes_after_long_onset"]], [500, 900])
        self.assertAlmostEqual(joint["episodes_after_long_onset"][1]["maximum_threshold_deficit"], .03)
        self.assertEqual(joint["final_target_streak"], {"start": 900, "end": 1000, "observations": 2, "sampled_span_steps": 100})
        self.assertEqual(joint["updates_since_last_failed_observation"], 200)
        windows = joint["fixed_tail_windows"]
        self.assertEqual([w["observations"] for w in windows], [2, 2, 3])
        self.assertEqual([w["failure_fraction"] for w in windows], [.5, .5, 1 / 3])
        self.assertAlmostEqual(joint["adjacent_complete_window_changes"][-1]["failure_fraction_change"], -1 / 6)

    def test_partial_and_empty_windows_do_not_create_frequency_trend(self):
        report = trajectory([0, .995, .995, .98, .97, .995, .995, .96, .96, .995, .995], through=700)
        result = describe(report, self.criterion(), window_steps=200)
        joint = result["metrics"]["joint"]
        self.assertFalse(result["complete_budget"])
        self.assertTrue(joint["episodes_after_long_onset"][-1]["ongoing_at_last_observation"])
        self.assertIsNone(joint["episodes_after_long_onset"][-1]["first_recovery"])
        self.assertEqual([w["complete_window"] for w in joint["fixed_tail_windows"]], [True, True, False])
        self.assertIsNone(joint["fixed_tail_windows"][-1]["failure_fraction"])
        self.assertEqual(len(joint["adjacent_complete_window_changes"]), 1)
        report = trajectory([0, .995, .995, .98, .97, .995, .995, .96, .96, .995, .995], through=600)
        joint = describe(report, self.criterion(), window_steps=200)["metrics"]["joint"]
        self.assertEqual([w["complete_window"] for w in joint["fixed_tail_windows"]], [True, False, False])
        self.assertEqual(joint["fixed_tail_windows"][1]["observations"], 1)
        self.assertIsNone(joint["fixed_tail_windows"][1]["episode_onsets_per_10000_updates"])
        self.assertEqual(joint["adjacent_complete_window_changes"], [])

    def test_train_only_failure_is_separate_from_heldout_recurrence(self):
        result = describe(trajectory([0, *([1] * 10)], train=[0, 1, 1, 1, 1, 1, 1, .9, 1, 1, 1]),
                          self.criterion(), window_steps=200)
        self.assertEqual(result["metrics"]["joint"]["episode_count"], 1)
        self.assertEqual(result["metrics"]["heldout"]["episode_count"], 0)
        self.assertEqual(result["metrics"]["joint"]["final_target_streak"]["observations"], 3)
        self.assertEqual(result["metrics"]["heldout"]["final_target_streak"]["observations"], 10)

    def test_absent_generalization_retains_failed_window_support(self):
        result = describe(trajectory([0, *([.1] * 10)]), self.criterion(), window_steps=200)
        self.assertIsNone(result["long_confirmation"])
        joint = result["metrics"]["joint"]
        self.assertIsNone(joint["episode_count"])
        self.assertIsNone(joint["post_long_onset_failure_fraction"])
        self.assertEqual([w["failure_fraction"] for w in joint["fixed_tail_windows"]], [1, 1, 1])

    def test_missing_or_invalid_observations_cannot_improve_frequency(self):
        original = trajectory([0, *([1] * 10)])
        for mutation in ("missing", "duplicate", "nonfinite", "final"):
            with self.subTest(mutation=mutation):
                report = deepcopy(original)
                if mutation == "missing":
                    del report["history"][6]
                elif mutation == "duplicate":
                    report["history"].insert(6, report["history"][6])
                elif mutation == "nonfinite":
                    report["history"][6]["heldout"]["accuracy"] = float("nan")
                else:
                    report["final"] = {**report["final"], "step": 0}
                with self.assertRaises(ValueError):
                    describe(report, self.criterion(), window_steps=200)

    def test_windows_cannot_claim_a_resolution_finer_than_canonical_sampling(self):
        report = trajectory([0, *([1] * 10)])
        for width in (0, 50, 123):
            with self.assertRaises(ValueError):
                describe(report, self.criterion(), window_steps=width)
        report["plan"]["config"]["eval_every"] = 0
        with self.assertRaises(ValueError):
            describe(report, self.criterion(), window_steps=200)
