"""Reading a study's runs back: their state, and the stability analyses of a modular run's records.

The records are those of the archived stability run of case `grokking` in
`fixtures/legacy_stability.json` and `fixtures/legacy_collapse.json`,
written into a run directory as training writes them; the report must read
them as the historical analyses did.
"""

import tempfile
import unittest

from lab.application.report import report_study
from lab.application.study import Study
from lab.domain.benchmarks import AssociativeRecall
from lab.domain.spec import describe, swap
from lab.infrastructure.store import RunDirectories

from examples import gptmini, modular
from test_collapse import CASES as COLLAPSE
from test_stability import CASES as STABILITY, archived

EXPERIMENT = modular(gptmini(), prime=193)
FINISHED = {"updates": 150000, "stop": {"step": 150000, "reason": "budget"}}


def history():
    """The archived run's canonical observations, each failure's held-out metrics and batch size whole."""
    failures = {point["step"]: point for point in COLLAPSE["grokking"]["failures"]}
    rows = archived("grokking")
    assert [row["heldout"]["accuracy"] for row in rows] == COLLAPSE["grokking"]["heldout"]
    return [{**row, **failures.get(row["step"], {})} for row in rows]


class ReportTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.runs = RunDirectories(directory.name)

    def write(self, label, experiment, rows, result=None):
        run = self.runs.open("stability", label)
        run.begin(describe(experiment), {"engine": {}}, "")
        for stream, records in (("history", rows), ("probes", COLLAPSE["grokking"]["probes"]),
                                ("diagnostics", COLLAPSE["grokking"]["diagnostics"])):
            for row in records:
                run.record(stream, row)
        if result is not None:
            run.finish(result)

    def report(self, experiments):
        return report_study(Study("stability", "", experiments), [], self.runs)

    def test_a_finished_modular_run_reads_as_the_stability_analyses_read_it(self):
        self.write("grokking", EXPERIMENT, history(), FINISHED)
        found = self.report({"grokking": EXPERIMENT, "pending": swap(EXPERIMENT, "optimizer.lr", 3e-3)})
        self.assertEqual(found["pending"], {"status": "not_started"})
        report = found["grokking"]
        self.assertEqual((report["status"], report["budget"], report["latest"]["step"], report["result"]),
                         ("finished", 150000, 150000, FINISHED))
        for key in ("phases", "persistence", "recovery"):
            self.assertEqual(report[key], STABILITY["grokking"][key])
        self.assertEqual(report["collapse"], COLLAPSE["grokking"]["collapse"])
        self.assertEqual(report["largest_gradients"], COLLAPSE["grokking"]["largest"])

    def test_only_a_run_that_reached_its_budget_is_read_as_complete(self):
        self.write("running", EXPERIMENT, history())
        self.write("stopped", EXPERIMENT, history(), {**FINISHED, "stop": {"step": 150000, "reason": "nonfinite"}})
        found = self.report({"running": EXPERIMENT, "stopped": EXPERIMENT})
        for label, status in (("running", "unfinished"), ("stopped", "stopped")):
            with self.subTest(label=label):
                self.assertEqual(found[label]["status"], status)
                self.assertFalse(found[label]["persistence"]["complete_canonical_history"])
                self.assertEqual(found[label]["recovery"]["through_update"], 150000)

    def test_recovery_reads_only_a_budget_that_spans_its_tail(self):
        short = swap(EXPERIMENT, "budget.updates", 20000)
        self.write("short", short, [row for row in history() if row["step"] <= 20000],
                   {"updates": 20000, "stop": {"step": 20000, "reason": "budget"}})
        report = self.report({"short": short})["short"]
        self.assertIn("nonnegative tail start", report["recovery"]["not_applicable"])
        self.assertTrue(report["persistence"]["complete_canonical_history"])
        self.assertEqual(report["persistence"]["tail_start_step"], 20000 - 50000)

    def test_a_run_is_read_against_the_file_that_defines_it(self):
        self.write("grokking", EXPERIMENT, history(), FINISHED)
        with self.assertRaisesRegex(ValueError, "different experiment; changed: optimizer.lr"):
            self.report({"grokking": swap(EXPERIMENT, "optimizer.lr", 3e-3)})
        raised = swap(EXPERIMENT, "budget.updates", 300000)
        self.assertEqual(self.report({"grokking": raised})["grokking"]["persistence"],
                         STABILITY["grokking"]["persistence"])

    def test_other_benchmarks_have_no_stability_analyses(self):
        recall = swap(EXPERIMENT, "benchmark", AssociativeRecall(length=16, vocab=64, alpha=.1, train=8,
                                                                 validation=8, test=8))
        self.write("recall", recall, [{"step": 0, "validation": {"accuracy": 0.0, "loss": 4.0}}])
        self.assertEqual(set(self.report({"recall": recall})["recall"]), {"status", "budget", "latest", "result"})


if __name__ == "__main__":
    unittest.main()
