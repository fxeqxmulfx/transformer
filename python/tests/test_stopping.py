"""`Selection` decides as the historical `EarlyStopper` did.

Golden decisions: the `stopper` sequences of `fixtures/legacy_text.json`,
random validation losses on a grid, so that ties and margins recur, observed
by `gpt_mini.domain.stopping.EarlyStopper` until it stops.
"""

import json
import math
from pathlib import Path
import unittest

from lab.domain.stopping import EarlyStopping, Selection

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "legacy_text.json").read_text())
REASONS = {"patience": "patience", "validation_divergence": "divergence", "nonfinite_validation": "nonfinite_selection"}


def policy(golden):
    return EarlyStopping(patience=golden["patience"], min_delta=golden["min_delta"], after=golden["min_steps"],
                         divergence=golden["divergence_delta"], divergence_patience=golden["divergence_patience"])


class SelectionTests(unittest.TestCase):
    def test_decisions_equal_the_historical_stopper(self):
        for group in FIXTURE["stopper"]:
            for index, sequence in enumerate(group["sequences"]):
                with self.subTest(policy=group["policy"], sequence=index):
                    selection = Selection(policy(group["policy"]))
                    for row in sequence:
                        self.assertIsNone(selection.stop)
                        new_best = selection.observe(row["step"], row["loss"])
                        stop = None if row["stop"] is None else (row["step"], REASONS[row["stop"]])
                        self.assertEqual((new_best, selection.step, selection.bad, selection.divergent, selection.stop),
                                         (row["new_best"], row["best_step"], row["bad"], row["divergent"], stop))

    def test_without_a_policy_only_the_best_is_kept(self):
        selection = Selection(None)
        self.assertEqual([selection.observe(step, loss) for step, loss in
                          enumerate([3.0, None, math.inf, 2.0, 2.0, 5.0, 1.5])],
                         [True, False, False, True, False, False, True])
        self.assertEqual((selection.best, selection.step, selection.stop), (1.5, 6, None))

    def test_no_stop_before_after(self):
        selection = Selection(EarlyStopping(patience=1, after=30, divergence_patience=1))
        for step in range(0, 30, 10):
            selection.observe(step, 1.0 if step == 0 else 2.0)
            self.assertIsNone(selection.stop)
        selection.observe(30, 2.0)
        self.assertEqual(selection.stop, (30, "divergence"))

    def test_a_nonfinite_loss_stops_at_once(self):
        selection = Selection(EarlyStopping(after=1000))
        selection.observe(0, 1.0)
        self.assertFalse(selection.observe(10, math.nan))
        self.assertEqual((selection.step, selection.stop), (0, (10, "nonfinite_selection")))

    def test_policies_are_checked(self):
        for fields in ({"patience": 0}, {"divergence_patience": 0}, {"after": -1}, {"min_delta": -1e-4},
                       {"min_delta": math.inf}, {"divergence": 0.0}):
            with self.subTest(**fields), self.assertRaises(ValueError):
                EarlyStopping(**fields)


if __name__ == "__main__":
    unittest.main()
