"""The collapse neighborhoods of archived stability runs, as the historical analysis read them.

Golden records: `fixtures/legacy_collapse.json`, the outputs of
`stability_analysis.collapse_neighborhoods` and the largest
post-confirmation gradients of `stability_analysis.summarize` of
`experiments/synthetic_trainers` at the commit the fixture names, on three
archived runs (AdamW and AMSGradW, with 5, 9 and 20 held-out failures after
generalizing), one that never generalizes, and one probe removed.
"""

import json
from pathlib import Path
import unittest

from lab.domain.collapse import collapse, largest_gradients
from lab.domain.stability import observed_updates

CASES = json.loads((Path(__file__).parent / "fixtures" / "legacy_collapse.json").read_text())["cases"]


def history(case):
    """The canonical observations of a case: the held-out accuracy of each, the whole of each failure."""
    failures = {point["step"]: point for point in case["failures"]}
    steps = observed_updates(case["config"]["steps"], case["config"]["eval_every"])
    assert len(steps) == len(case["heldout"])
    return [failures.get(step, {"step": step, "heldout": {"accuracy": accuracy}})
            for step, accuracy in zip(steps, case["heldout"], strict=True)]


class HistoricalCollapseTests(unittest.TestCase):
    def test_every_archived_case_reads_as_the_historical_analysis_read_it(self):
        for name, case in CASES.items():
            if "failures" not in case:
                continue
            with self.subTest(case=name):
                found = collapse(history(case), case["probes"], case["diagnostics"],
                                 case["config"]["target"], case["config"]["patience"])
                self.assertEqual(found, case["collapse"])
                self.assertEqual(largest_gradients(case["diagnostics"], found["confirmed_step"]), case["largest"])

    def test_a_failure_without_its_next_probe_has_no_neighborhood(self):
        source, case = CASES["grokking"], CASES["unsupported"]
        probes = [point for point in source["probes"] if point["step"] != case["dropped_probe"]]
        self.assertEqual(collapse(history(source), probes, source["diagnostics"]), case["collapse"])
        self.assertEqual(case["collapse"]["missing_neighbor_support"], 1)

    def test_the_cases_cover_every_category_and_both_moment_records(self):
        found = {name: case["collapse"] for name, case in CASES.items()}
        self.assertGreater(found["isolated"]["isolated_at_canonical_observation"], 0)
        self.assertGreater(found["grokking"]["recovered_by_next_update"], 0)
        self.assertGreater(found["raw"]["still_below_target_both_neighbors"], 0)
        self.assertEqual(found["plateau"]["scope"], "no_confirmed_heldout_event")
        moments = {name: {key for key in CASES[name]["largest"][0] if key.endswith("_l2")} for name in ("grokking", "raw")}
        self.assertLessEqual({"exp_avg_l2", "exp_avg_sq_l2"}, moments["grokking"])
        self.assertLessEqual({"m_l2", "v_l2", "maximum_l2"}, moments["raw"])
        self.assertIn("maximum_buffer_max", CASES["raw"]["largest"][0])


if __name__ == "__main__":
    unittest.main()
