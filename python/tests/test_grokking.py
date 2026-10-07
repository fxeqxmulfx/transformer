"""Causal phase evidence, including immediate learning and false loss alarms."""

import math
import unittest

from lab.domain.grokking import GrokkingDiagnostics, ProgressCriterion, available, norm_progress, progress
from lab.domain.spec import swap

from examples import gptmini, modular


def history(count=100, immediate=False, structure=False):
    points = []
    for i in range(count):
        step = 250 * i
        row = {"step": step,
               "train": {"answer_accuracy": 1., "answer_loss": .001},
               "heldout": {"answer_accuracy": 1. if immediate else .01,
                           "answer_loss": 5 * math.exp(-step / 10_000)}}
        if structure and step % 1000 == 0:
            row["grokking"] = {
                "heldout_invariant_energy_fraction": min(.99, .02 + step / 50_000),
                "heldout_restricted_loss": 5 * math.exp(-step / 5000),
                "heldout_residual_energy": math.exp(-step / 10_000)}
        points.append(row)
    return points


class GrokkingProgressTests(unittest.TestCase):
    def test_accuracy_only_archives_and_actual_optimizer_norms(self):
        self.assertFalse(available([{"step": 0, "train": {"accuracy": 0}, "heldout": {"accuracy": 0}}]))
        self.assertTrue(available(history()))
        rows = [{"step": 1000, "parameters": {
            "embed.weight": {"parameter_l2": 3., "parameter_before_l2": 6., "gradient_l2": .6, "update_l2": .03},
            "blocks.0.attention.output.weight": {
                "parameter_l2": 4., "parameter_before_l2": 8., "gradient_l2": .8, "update_l2": .04}}}]
        found = norm_progress(rows)["latest"]["groups"]
        self.assertEqual(found["all"]["parameter_l2"], 5.)
        self.assertEqual(found["all"]["gradient_l2"], 1.)
        self.assertAlmostEqual(found["all"]["relative_update_l2"], .005)
        self.assertEqual(found["attention"]["parameter_l2"], 4.)

    def test_adding_future_observations_cannot_rewrite_a_signal(self):
        rows = history(structure=True)
        full = progress(rows)
        self.assertIsNotNone(full["first_events"]["structure_forming"])
        for length in (1, 5, 21, 35, 66, 99):
            before = progress(rows[:length])
            self.assertEqual(before["observations"], full["observations"][:length])
            for event, first in before["first_events"].items():
                expected = full["first_events"][event]
                self.assertEqual(first, expected if expected is not None and expected <= rows[length - 1]["step"] else None)

    def test_loss_drop_without_generalization_is_not_structure_evidence(self):
        result = progress(history())
        self.assertIsNotNone(result["first_events"]["loss_only_alarm"])
        self.assertIsNone(result["first_events"]["structure_forming"])
        self.assertIsNone(result["first_events"]["observed_delayed_generalization"])
        self.assertEqual(result["structure_observations"], 0)

    def test_immediate_generalization_is_not_reported_as_grokking(self):
        result = progress(history(immediate=True, structure=True))
        self.assertEqual(result["latest"]["phase"], "generalized")
        self.assertTrue(all(first is None for first in result["first_events"].values()))

    def test_delayed_generalization_requires_confirmed_fit_and_accuracy(self):
        rows = history(50, structure=True)
        for point in rows[40:]:
            point["heldout"]["answer_accuracy"] = 1.
        result = progress(rows)
        self.assertEqual(result["first_events"]["observed_delayed_generalization"], 11_000)
        self.assertEqual(result["latest"]["phase"], "generalized")

    def test_symmetric_wrong_rule_and_constant_logits_do_not_trigger_formation(self):
        rows = history(structure=True)
        for row in rows:
            if "grokking" in row:
                row["grokking"]["heldout_invariant_energy_fraction"] = 1.
                row["grokking"]["heldout_restricted_loss"] = 10.
        result = progress(rows)
        self.assertIsNone(result["first_events"]["structure_forming"])
        self.assertIsNone(result["first_events"]["cleanup"])
        for row in rows:
            if "grokking" in row:
                row["grokking"]["heldout_invariant_energy_fraction"] = None
        self.assertIsNone(progress(rows)["first_events"]["structure_forming"])

    def test_rejects_invalid_time_losses_and_criterion(self):
        for rows in ([history()[0], history()[0]],
                     [{**history()[0], "step": .5}],
                     [{**history()[0], "heldout": {"answer_accuracy": .1, "answer_loss": float("nan")}}]):
            with self.assertRaises(ValueError):
                progress(rows)
        for settings in ({"loss_window": 5}, {"patience": 1}, {"structure_window": 3},
                         {"minimum_delay": 0}, {"cleanup_share": 1.1}):
            with self.assertRaises(ValueError):
                ProgressCriterion(**settings)

    def test_diagnostics_reject_incompatible_cadence_and_domain(self):
        base = modular(gptmini(16, 2, 2), prime=11, every=4)
        swap(base, "diagnostics", GrokkingDiagnostics(orbit_every=8))
        with self.assertRaises(ValueError):
            swap(base, "diagnostics", GrokkingDiagnostics(orbit_every=6))
        with self.assertRaises(ValueError):
            swap(swap(base, "benchmark.prime", 2), "diagnostics", GrokkingDiagnostics(orbit_every=8))
        for settings in ({"batch": 0}, {"orbit_every": 1.5}, {"energy_floor": float("inf")}):
            with self.assertRaises(ValueError):
                GrokkingDiagnostics(**settings)
