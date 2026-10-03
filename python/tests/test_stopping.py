"""`Selection` decides as the historical `EarlyStopper` did.

Golden decisions: the `stopper` sequences of `fixtures/legacy_text.json`,
random validation losses on a grid, so that ties and margins recur, observed
by `gpt_mini.domain.stopping.EarlyStopper` until it stops.
"""

import json
import math
from pathlib import Path
import unittest

from examples import gptmini, modular
from lab.dsl import Parity, Solved, Synthetic, TinyShakespeare, swap
from lab.domain.stopping import EarlyStopping, Selection

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "legacy_text.json").read_text())
REASONS = {"patience": "patience", "validation_divergence": "divergence", "nonfinite_validation": "nonfinite_selection"}


def by_loss(metrics):
    return -metrics["loss"]


def losses(selection, observations):
    """Observe each (step, loss); whether each was the best so far."""
    return [selection.observe(step, {"loss": loss}) for step, loss in observations]


def policy(golden):
    return EarlyStopping(patience=golden["patience"], min_delta=golden["min_delta"], after=golden["min_steps"],
                         divergence=golden["divergence_delta"], divergence_patience=golden["divergence_patience"])


class SelectionTests(unittest.TestCase):
    def test_decisions_equal_the_historical_stopper(self):
        for group in FIXTURE["stopper"]:
            for index, sequence in enumerate(group["sequences"]):
                with self.subTest(policy=group["policy"], sequence=index):
                    selection = Selection(policy(group["policy"]), by_loss)
                    for row in sequence:
                        self.assertIsNone(selection.stop)
                        new_best = selection.observe(row["step"], {"loss": row["loss"]})
                        stop = None if row["stop"] is None else (row["step"], REASONS[row["stop"]])
                        self.assertEqual((new_best, selection.step, selection.bad, selection.divergent, selection.stop),
                                         (row["new_best"], row["best_step"], row["bad"], row["divergent"], stop))

    def test_without_a_policy_only_the_best_is_kept(self):
        selection = Selection(None, by_loss)
        self.assertEqual(losses(selection, enumerate([3.0, None, math.inf, 2.0, 2.0, 5.0, 1.5])),
                         [True, False, False, True, False, False, True])
        self.assertEqual((selection.rank, selection.step, selection.stop), (-1.5, 6, None))

    def test_the_rank_selects_and_the_policy_watches_the_loss(self):
        # Accuracy first, then the loss: the best is the first of the highest rank.
        selection = Selection(EarlyStopping(patience=3, after=0, divergence=1.0, divergence_patience=9),
                              lambda metrics: (metrics["accuracy"], -metrics["loss"]))
        observations = [(0.5, 1.0), (0.5, 0.8), (0.9, 2.0), (0.9, 2.0), (0.9, 1.9)]
        self.assertEqual([selection.observe(step, {"accuracy": accuracy, "loss": loss})
                          for step, (accuracy, loss) in enumerate(observations)],
                         [True, True, True, False, True])
        # The loss last fell at update 1, to 0.8; from there 1.8 or more diverges.
        self.assertEqual((selection.step, selection.lowest, selection.divergent, selection.stop),
                         (4, 0.8, 3, (4, "patience")))

    def test_no_stop_before_after(self):
        selection = Selection(EarlyStopping(patience=1, after=30, divergence_patience=1), by_loss)
        for step in range(0, 30, 10):
            losses(selection, [(step, 1.0 if step == 0 else 2.0)])
            self.assertIsNone(selection.stop)
        losses(selection, [(30, 2.0)])
        self.assertEqual(selection.stop, (30, "divergence"))

    def test_a_nonfinite_loss_stops_at_once(self):
        selection = Selection(EarlyStopping(after=1000), by_loss)
        self.assertEqual(losses(selection, [(0, 1.0), (10, math.nan)]), [True, False])
        self.assertEqual((selection.step, selection.stop), (0, (10, "nonfinite_selection")))

    def test_solved_stops_at_the_first_observation_that_reaches_the_target(self):
        benchmark = Synthetic(task=Parity(scratchpad="running"), length=8, target=0.9)
        selection = Selection(Solved(), benchmark.rank, benchmark.solved)
        for step, accuracy, loss in ((0, 0.25, 2.0), (10, 0.875, 0.5), (20, 0.9, 0.6)):
            self.assertIsNone(selection.stop)
            selection.observe(step, {"sequence_accuracy": accuracy, "balanced_accuracy": accuracy, "loss": loss})
        self.assertEqual((selection.step, selection.stop), (20, (20, "solved")))
        selection = Selection(Solved(), benchmark.rank, benchmark.solved)
        self.assertEqual(losses(selection, [(0, math.nan)]), [False])
        self.assertEqual(selection.stop, (0, "nonfinite_selection"))

    def test_only_a_benchmark_with_a_target_stops_when_solved(self):
        text = swap(modular(gptmini()), "benchmark", TinyShakespeare(window=50))
        with self.assertRaisesRegex(ValueError, "TinyShakespeare sets no target"):
            swap(text, "stopping", Solved())

    def test_policies_are_checked(self):
        for fields in ({"patience": 0}, {"divergence_patience": 0}, {"after": -1}, {"min_delta": -1e-4},
                       {"min_delta": math.inf}, {"divergence": 0.0}):
            with self.subTest(**fields), self.assertRaises(ValueError):
                EarlyStopping(**fields)


if __name__ == "__main__":
    unittest.main()
