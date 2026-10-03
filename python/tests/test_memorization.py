"""What a memorization study's history shows: fits, generalization on novel inputs, transfer, and curves.

The cases of `experiments/synthetic_trainers/tests/test_study_metrics.py`
and `test_study_data.py`, on the observations as the trainer now records them.
"""

import unittest

from lab.domain.generative import Copy, RandomLM
from lab.domain.memorization import Memorization, delayed_generalization, fit_value, report
from lab.domain.spec import swap
from lab.domain.synthetic import Synthetic

BENCHMARK = Synthetic(task=Copy(), length=4, ood=(8,), study=Memorization(), target=.9)


def scores(accuracy):
    return {"sequence_accuracy": accuracy, "token_accuracy": accuracy, "example_loss": 1 - accuracy}


def history(held_out=(0, .1, 1, .3, 1, 1)):
    """The training split fits its oracle's labels at step 5 and its observed ones at 10; validation is perfect."""
    return [{"step": step, "training_seconds": step / 2, "epochs_seen": step / 10,
             "train": scores(observed), "train_clean": scores(clean), "validation": scores(1),
             "validation/novel": scores(1), "validation/length-8": scores(probe),
             "validation/length-8/novel": scores(probe)}
            for step, clean, observed, probe in zip((0, 5, 10, 20, 30, 40), (0, 1, 1, 1, 1, 1),
                                                    (0, .5, 1, 1, 1, 1), held_out)]


def overwrite(rows, split, metrics):
    return [{**row, split: metrics(row)} for row in rows]


class GeneralizationTests(unittest.TestCase):
    def test_transfer_lag_requires_stable_held_out_success_and_records_both_train_fits(self):
        found = delayed_generalization(history(), BENCHMARK)
        self.assertEqual((found["clean_train_fit_step"], found["observed_train_fit_step"]), (5, 10))
        self.assertEqual((found["generalization_step"], found["confirmed_at_step"]), (30, 40))
        self.assertEqual((found["lag_steps"], found["lag_epochs"], found["lag_training_seconds"]), (25, 2.5, 12.5))
        self.assertTrue(found["delayed_transfer_candidate"])

    def test_generalization_in_distribution_is_separate_from_transfer(self):
        rows = overwrite(history(held_out=(.5,) * 6), "validation/novel", lambda row: scores(float(row["step"] >= 20)))
        found = delayed_generalization(rows, BENCHMARK)
        self.assertEqual((found["id_generalization_step"], found["id_confirmed_at_step"], found["id_lag_steps"]),
                         (20, 30, 15))
        self.assertTrue(found["delayed_id_generalization_candidate"])
        self.assertIsNone(found["generalization_step"])
        self.assertFalse(found["delayed_transfer_candidate"])

    def test_generalization_needs_novel_inputs_and_an_answer_to_generalize_to(self):
        control = Synthetic(task=RandomLM(), length=4, ood=(8,), study=Memorization(), target=.9)
        self.assertIsNone(delayed_generalization(history(), control)["id_generalization_step"])
        repeated = overwrite(history(), "validation/novel", lambda row: None)
        self.assertIsNone(delayed_generalization(repeated, BENCHMARK)["id_generalization_step"])
        self.assertIsNone(delayed_generalization(repeated, BENCHMARK)["generalization_step"])

    def test_transfer_needs_every_held_out_distribution_and_enough_observations(self):
        self.assertIsNone(delayed_generalization(history()[:-1], BENCHMARK)["generalization_step"])
        for benchmark in (swap(BENCHMARK, "ood", ()), swap(BENCHMARK, "target", None)):
            with self.subTest(benchmark=benchmark):
                self.assertIsNone(delayed_generalization(history(), benchmark)["generalization_step"])
        two = swap(BENCHMARK, "ood", (8, 12))
        rows = overwrite(history(), "validation/length-12/novel", lambda row: None)
        self.assertIsNone(delayed_generalization(rows, two)["generalization_step"])
        rows = overwrite(rows, "validation/length-12/novel", lambda row: row["validation/length-8/novel"])
        self.assertEqual(delayed_generalization(rows, two)["generalization_step"], 30)

    def test_the_fit_is_the_error_or_the_loss_of_teacher_forcing(self):
        metrics = {"sequence_accuracy": 1, "token_accuracy": .75, "example_loss": .2}
        self.assertEqual([fit_value(metrics, fit) for fit in ("example_error", "token_error", "loss")], [0, .25, .2])
        self.assertEqual(delayed_generalization(history(), swap(BENCHMARK, "study", Memorization(fit="loss")))[
            "clean_train_fit_step"], 5)


class ReportTests(unittest.TestCase):
    def test_each_curve_is_searched_and_each_peak_is_set_beside_the_transfer(self):
        found = report(history(), BENCHMARK)
        witnesses = found["epoch_double_descent"]
        self.assertEqual(set(witnesses), {"validation_example_loss", "validation_error", "validation_novel_error",
                                          "length-8/example_loss", "length-8/error", "length-8/novel_error"})
        self.assertIsNone(witnesses["validation_error"])
        self.assertEqual(witnesses["length-8/error"]["indices"], [0, 2, 3, 5])
        self.assertEqual(found["peak_transition_alignment"], {
            name: {"peak_step": 20, "generalization_step": 30, "transition_after_peak": True,
                   "scope": "temporal_alignment_only; not_a_causal_test"}
            for name in ("length-8/example_loss", "length-8/error", "length-8/novel_error")})
        interpolation = found["interpolation"]
        self.assertEqual((interpolation["first_observed_step"], interpolation["final_value"],
                          interpolation["final_fitted"]), (10, 0, True))

    def test_a_split_without_novel_inputs_has_no_novel_curve(self):
        rows = overwrite(history(), "validation/length-8/novel", lambda row: None)
        self.assertNotIn("length-8/novel_error", report(rows, BENCHMARK)["epoch_double_descent"])


class MemorizationSpecTests(unittest.TestCase):
    def test_invalid_study_controls_are_rejected(self):
        for fields in ({"noise": float("nan")}, {"noise": 1.5}, {"noise_seed": -1}, {"pool": 0}, {"fit": "error"},
                       {"epsilon": 0}, {"epsilon": 1}, {"patience": 0}, {"tolerance": -1},
                       {"tolerance": float("inf")}):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                Memorization(**fields)


if __name__ == "__main__":
    unittest.main()
