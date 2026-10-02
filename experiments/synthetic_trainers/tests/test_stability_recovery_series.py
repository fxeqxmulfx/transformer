"""Check whole-history windows without hiding cross-window or partial support."""

import unittest

from experiments.synthetic_trainers.persistence import PersistenceConfig
from experiments.synthetic_trainers.stability_recovery_series import timeline
from experiments.synthetic_trainers.tests.test_stability_recovery import trajectory


class RecoverySeriesTests(unittest.TestCase):
    criterion = PersistenceConfig(confirmation_observations=2, tail_steps=600)

    def test_all_post_onset_support_and_boundary_crossing_episode_retained(self):
        report = trajectory([0, .995, .995, .98, .97, .995, .995, .96, .96, .995, .995])
        joint = timeline(report, self.criterion, window_steps=200)["metrics"]["joint"]
        leading = joint["leading_partial_width_window"]
        self.assertEqual((leading["start"], leading["stop"], leading["observations"]), (100, 200, 1))
        self.assertIsNone(leading["episode_onsets_per_10000_updates"])
        windows = joint["fixed_grid_windows"]
        self.assertEqual([w["observations"] for w in windows], [2, 2, 2, 3])
        self.assertEqual([w["failed_observations"] for w in windows], [1, 1, 1, 1])
        self.assertEqual([w["new_episode_onsets"] for w in windows], [1, 0, 1, 0])
        self.assertEqual(len(joint["adjacent_complete_nominal_window_changes"]), 3)

    def test_future_and_partial_windows_cannot_appear_as_completed_zero_rates(self):
        report = trajectory([0, .995, .995, .98, .97, .995, .995, .96, .96, .995, .995], through=600)
        joint = timeline(report, self.criterion, window_steps=200)["metrics"]["joint"]
        windows = joint["fixed_grid_windows"]
        self.assertEqual([w["observations"] for w in windows], [2, 2, 1, 0])
        self.assertEqual([w["complete_window"] for w in windows], [True, True, False, False])
        self.assertIsNone(windows[2]["episode_onsets_per_10000_updates"])
        self.assertIsNone(windows[3]["failure_fraction"])
        self.assertEqual(len(joint["adjacent_complete_nominal_window_changes"]), 1)

    def test_absent_confirmation_and_exact_grid_onset(self):
        self.assertIsNone(timeline(trajectory([0, *([.1] * 10)]), self.criterion, window_steps=200)["metrics"])
        report = trajectory([0, 0, *([1] * 9)])
        joint = timeline(report, self.criterion, window_steps=200)["metrics"]["joint"]
        self.assertIsNone(joint["leading_partial_width_window"])
        self.assertEqual(sum(w["observations"] for w in joint["fixed_grid_windows"]), 9)
        self.assertEqual(sum(w["failed_observations"] for w in joint["fixed_grid_windows"]), 0)
