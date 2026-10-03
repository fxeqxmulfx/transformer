"""The phases, persistence and recovery of archived stability runs, as the historical analyses read them.

Golden records: `fixtures/legacy_stability.json`, the outputs of
`paper_phases.diagnose`, `persistence.assess` and `stability_recovery.describe`
of `experiments/synthetic_trainers` at the commit the fixture names, on the
canonical histories of three archived 150,000-update runs (a plateau that
groks, a plateau that never does, a run that fails in 21 episodes) and on
two histories derived from them: one that persists, one stopped short.
"""

import copy
import json
from pathlib import Path
import unittest

from lab.domain.phases import phases, sustained
from lab.domain.stability import Persistence, observed_updates, persistence, recovery

CASES = json.loads((Path(__file__).parent / "fixtures" / "legacy_stability.json").read_text())["cases"]


def archived(name):
    """The observations of an archived case: accuracies and cumulative seconds at each canonical update."""
    case = CASES[name]
    columns = case["columns"]
    steps = observed_updates(case["config"]["steps"], case["config"]["eval_every"])
    assert len(steps) == len(columns["train"])
    return [{"step": step, "training_seconds": columns["training_seconds"][index],
             "wall_seconds": columns["wall_seconds"][index], "train": {"accuracy": columns["train"][index]},
             "heldout": {"accuracy": columns["heldout"][index]}} for index, step in enumerate(steps)]


def generalized(history):
    """`history` with every observation from its first sustained held-out success on at accuracy 1."""
    onset = sustained(history, "heldout", .99, 2)["onset"]
    changed = copy.deepcopy(history)
    for point in changed:
        if point["step"] >= onset:
            point["train"]["accuracy"] = point["heldout"]["accuracy"] = 1.0
    return changed


def observations(name):
    if name == "persistent":
        return generalized(archived("grokking"))
    if name == "unfinished":
        return [point for point in archived("unstable") if point["step"] <= 100000]
    return archived(name)


class HistoricalStabilityTests(unittest.TestCase):
    def test_every_case_reads_as_the_historical_analyses_read_it(self):
        for name, case in CASES.items():
            with self.subTest(case=name):
                history = observations(name)
                budget, every, finished = case["config"]["steps"], case["config"]["eval_every"], case["finished"]
                self.assertEqual(phases(history), case["phases"])
                self.assertEqual(persistence(history, budget, every, finished), case["persistence"])
                self.assertEqual(recovery(history, budget, every, finished), case["recovery"])

    def test_the_cases_cover_both_outcomes_of_each_criterion(self):
        persisted = {name: case["persistence"]["persistent_final_performance"] for name, case in CASES.items()}
        self.assertEqual(persisted, {"grokking": False, "plateau": False, "unstable": False, "persistent": True,
                                     "unfinished": False})
        self.assertTrue(CASES["persistent"]["persistence"]["stable_grokking"])
        self.assertTrue(CASES["grokking"]["phases"]["observed_plateau_then_generalization"])
        self.assertIsNone(CASES["plateau"]["phases"]["sustained_heldout_target"])


class RecoveryInputTests(unittest.TestCase):
    def test_recovery_reads_only_an_exact_canonical_prefix(self):
        history = archived("unstable")
        with self.assertRaisesRegex(ValueError, "canonical prefix"):
            recovery(history[:100] + history[101:], 150000, 250, finished=False)
        with self.assertRaisesRegex(ValueError, "canonical prefix"):
            recovery(history[:-1], 150000, 250, finished=True)
        with self.assertRaisesRegex(ValueError, "cadence multiple"):
            recovery(history, 150000, 250, finished=True, window=1000 + 125)
        with self.assertRaisesRegex(ValueError, "finite"):
            broken = copy.deepcopy(history)
            broken[5]["heldout"]["accuracy"] = float("nan")
            recovery(broken, 150000, 250, finished=True)

    def test_a_criterion_orders_its_thresholds_and_counts_positive_spans(self):
        with self.assertRaisesRegex(ValueError, "thresholds"):
            Persistence(target=.1, heldout_ceiling=.1)
        with self.assertRaisesRegex(ValueError, "positive"):
            Persistence(tail_steps=0)


if __name__ == "__main__":
    unittest.main()
