import unittest

from lab.domain import cadence
from lab.domain.analysis import curve_witness, milestones, transition
from lab.domain.spec import swap
from lab.domain.training import Cosine, Diagnostics, Schedule, rate

from examples import gptmini, modular


class RateTests(unittest.TestCase):
    """Golden values from the historical trainers' own functions (commit 2c81c3f)."""

    def test_warmup_matches_grokking_learning_rate(self):
        expected = {0: "0x0.0p+0", 1: "0x1.a36e2eb1c432dp-14", 3: "0x1.3a92a30553261p-12",
                    7: "0x1.6f0068db8bac7p-11", 9: "0x1.d7dbf487fcb93p-11", 10: "0x1.0624dd2f1a9fcp-10",
                    11: "0x1.0624dd2f1a9fcp-10", 1000: "0x1.0624dd2f1a9fcp-10"}
        for completed, value in expected.items():
            self.assertEqual(rate(.001, Schedule(warmup=10), completed).hex(), value)
        for completed in (0, 5):
            self.assertEqual(rate(.0003, Schedule(), completed).hex(), "0x1.3a92a30553261p-12")

    def test_cosine_tail_matches_scheduled_expected_rate(self):
        schedule = Schedule(warmup=10, anneal=Cosine(start=150_000, end=250_000, final=.1))
        expected = {0: "0x0.0p+0", 3: "0x1.797cc39ffd60ep-14", 10: "0x1.3a92a30553261p-12",
                    150_000: "0x1.3a92a30553261p-12", 150_001: "0x1.3a92a304271eap-12",
                    175_000: "0x1.111c8abdcb9f1p-12", 199_999: "0x1.5a09fa38dc1e5p-13",
                    237_777: "0x1.4e21502d8c7c4p-15", 250_000: "0x1.f75104d551d68p-16",
                    260_000: "0x1.f75104d551d68p-16"}
        for completed, value in expected.items():
            self.assertEqual(rate(.0003, schedule, completed).hex(), value)

    def test_inclusive_warmup_matches_the_convex_mqar_trainer(self):
        # `train_run` at length 64 of the full profile: 64 epochs of 1563 updates, warmup over 10003.
        expected = {0: "0x1.ce85c72575799p-23", 1: "0x1.ce85c72575799p-22", 4999: "0x1.1a4d26cc9cf37p-10",
                    10_001: "0x1.1a5b9afad61f2p-9", 10_002: "0x1.1a62d511f2b4fp-9",
                    10_003: "0x1.1a62d511f2b4fp-9", 100_031: "0x1.1a62d511f2b4fp-9"}
        for completed, value in expected.items():
            self.assertEqual(rate(0.002154434690031882, Schedule(warmup=10_003, inclusive=True), completed).hex(),
                             value)

    def test_annealing_is_checked_against_warmup_and_budget(self):
        with self.assertRaisesRegex(ValueError, "after warmup"):
            Schedule(warmup=20, anneal=Cosine(start=10, end=30, final=.1))
        with self.assertRaisesRegex(ValueError, "within the budget"):
            swap(modular(gptmini(), updates=100), "schedule.anneal", Cosine(start=50, end=200, final=.1))


class CadenceTests(unittest.TestCase):
    def test_sampling_matches_the_historical_diagnostics_config(self):
        experiment = swap(modular(gptmini(), updates=40, every=10), "diagnostics",
                          Diagnostics(every=4, neighbors=True))
        steps = range(1, 40)
        self.assertEqual([s for s in steps if cadence.sampled(experiment, s)],
                         [1, 4, 8, 9, 11, 12, 16, 19, 20, 21, 24, 28, 29, 31, 32, 36, 39])
        self.assertEqual([s for s in steps if cadence.probe(experiment, s)],
                         [1, 9, 11, 19, 21, 29, 31, 39])

    def test_canonical_observations_include_the_last_update(self):
        experiment = modular(gptmini(), updates=12_345, every=250)
        self.assertTrue(cadence.canonical(experiment, 12_345))
        self.assertTrue(cadence.checkpointed(experiment, 10_000))
        self.assertTrue(cadence.checkpointed(experiment, 12_345))
        self.assertFalse(cadence.checkpointed(experiment, 12_250))


class AnalysisTests(unittest.TestCase):
    def test_transition_needs_a_sustained_heldout_streak(self):
        row = lambda step, train, heldout: {"step": step, "train": {"accuracy": train},
                                            "heldout": {"accuracy": heldout}}
        history = [row(0, 0, 0), row(10, 1, .5), row(20, 1, .995), row(30, 1, .2),
                   row(40, 1, .999), row(50, 1, 1)]
        result = transition(history)
        self.assertEqual((result["train_fit_step"], result["heldout_onset_step"],
                          result["heldout_confirmed_step"], result["lag_steps"]), (10, 40, 50, 30))
        self.assertTrue(result["delayed_generalization"])

    def test_milestones_record_the_first_crossing_and_whether_it_held(self):
        row = lambda step, correct: {"step": step, "validation": {"accuracy": correct / 8, "correct": correct,
                                                                  "queries": 8}}
        found = milestones([row(0, 6), row(5, 4), row(10, 8), row(15, 7)], "validation", (50, 75, 90, 100))
        # Four of eight reach 50% exactly.
        self.assertEqual(found["50"], {"step": 0, "accuracy": .75, "previous_step": None, "previous_accuracy": None,
                                       "sustained_to_end": True})
        self.assertEqual(found["75"], {"step": 0, "accuracy": .75, "previous_step": None, "previous_accuracy": None,
                                       "sustained_to_end": False})
        self.assertEqual(found["90"], {"step": 10, "accuracy": 1.0, "previous_step": 5, "previous_accuracy": .5,
                                       "sustained_to_end": False})
        self.assertEqual(found["100"], found["90"])
        self.assertIsNone(milestones([row(0, 3)], "validation", (50,))["50"])

    def test_curve_witness_finds_descent_ascent_descent(self):
        self.assertEqual(curve_witness([(0, 3), (1, 1), (2, 2), (3, 0)])["indices"], [0, 1, 2, 3])
        self.assertIsNone(curve_witness([(0, 3), (1, 2), (2, 1), (3, 0)]))
        self.assertIsNone(curve_witness([(0, 3), (1, 1), (2, 1.01), (3, 0)], .02))


if __name__ == "__main__":
    unittest.main()
