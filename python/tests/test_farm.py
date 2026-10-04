"""Runs side by side record what each records alone; they start device first, then widest, on cores by package."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

from lab.domain.training import Compiled, CudaGraph
from lab.infrastructure.farm import Cores, order
from lab.infrastructure.store import STREAMS, RunDirectory
from lab.interfaces.cli import main

from test_cli import PAIR
from test_engine import untimed


class CoresTests(unittest.TestCase):
    def test_a_run_takes_the_fullest_package_that_holds_it(self):
        cores = Cores({0: [(0, 4), (1, 5)], 1: [(2, 6), (3, 7), (8, 9)]})
        self.assertEqual(cores.take(2), [(0, 4), (1, 5)])
        self.assertEqual(cores.take(1), [(2, 6)])
        self.assertIsNone(cores.take(3))
        cores.give([(0, 4), (1, 5)])
        self.assertEqual(cores.take(2), [(0, 4), (1, 5)])
        self.assertEqual(cores.take(2), [(3, 7), (8, 9)])
        self.assertIsNone(cores.take(1))

    def test_a_run_holding_a_device_starts_first_then_the_widest(self):
        runs = [("one", Compiled()), ("four", Compiled(threads=4)), ("graph", CudaGraph()), ("other", Compiled()),
                ("two", Compiled(threads=2))]
        self.assertEqual([label for label, _ in order(runs)], ["graph", "four", "two", "one", "other"])

    def test_only_a_run_wider_than_every_package_spans_packages(self):
        cores = Cores({0: [(0,), (1,)], 1: [(2,), (3,)]})
        self.assertEqual(cores.take(1), [(0,)])
        self.assertIsNone(cores.take(4))
        self.assertEqual(cores.take(3), [(2,), (3,), (1,)])
        cores.give([(2,), (3,), (1,)])
        self.assertEqual(cores.free, {0: [(1,)], 1: [(2,), (3,)]})


class SideBySideTests(unittest.TestCase):
    def experiment(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        folder = Path(directory.name) / "pair"
        folder.mkdir()
        (folder / "experiment.py").write_text(PAIR)
        return folder

    def run_lab(self, *arguments):
        stream = io.StringIO()
        with redirect_stdout(stream):
            self.assertEqual(main(["run", *map(str, arguments)]), 0)
        return stream.getvalue().splitlines()

    def test_runs_side_by_side_record_what_they_record_alone(self):
        apart, alone = self.experiment(), self.experiment()
        lines = self.run_lab(apart)
        for label in ("softmax", "sparsemax"):
            self.assertIn([label, "step", "20"], [line.split()[:3] for line in lines])
            self.assertEqual(sum(line.startswith(f"{label} finished 20 updates") for line in lines), 2)
            self.run_lab(alone, label)
            together, single = RunDirectory(apart / "runs" / label), RunDirectory(alone / "runs" / label)
            for stream in STREAMS:
                self.assertEqual(untimed(together.records(stream)), untimed(single.records(stream)))
            self.assertEqual(json.dumps(together.result()["final"]["heldout"]),
                             json.dumps(single.result()["final"]["heldout"]))
        self.assertEqual(self.run_lab(apart), [line for line in lines if " finished " in line][-2:])


if __name__ == "__main__":
    unittest.main()
