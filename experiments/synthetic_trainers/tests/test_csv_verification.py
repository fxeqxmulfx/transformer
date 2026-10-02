"""CSV runtime repair retains frozen outcomes and bounds dense-trace work."""

import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from experiments.synthetic_trainers import stability_report
from experiments.synthetic_trainers.protocols.adamw_stability_20261002.csv_verification import (
    linear_csv_verification, verify_csv)


class CsvVerificationTests(unittest.TestCase):
    def test_equivalent_cells_and_corruption_rejection(self):
        samples = [[], [{"a": 1, "b": None}],
                   [{"a": 1, "b": None}, {"a": -2.5, "c": True}, {"b": "", "c": "x,y\nz"}]]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "table.csv"
            for rows in samples:
                with self.subTest(rows=rows):
                    stability_report.csv_rows(path, rows)
                    stability_report.verify_csv(path, rows)
                    verify_csv(path, rows)
                    path.write_text(path.read_text() + "corrupt\nvalue\n")
                    for function in (stability_report.verify_csv, verify_csv):
                        with self.assertRaisesRegex(ValueError, "differs from measured rows"):
                            function(path, rows)

    def test_large_table_traversal_stays_linear(self):
        class CountedRows(list):
            traversals = 0

            def __iter__(self):
                self.traversals += 1
                return super().__iter__()

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "trace.csv"
            rows = CountedRows({"step": step, "gradient_l2": step / 1000, "epoch_tail": False}
                               for step in range(1, 150001))
            stability_report.csv_rows(path, rows)
            rows.traversals = 0
            verify_csv(path, rows)
            self.assertEqual(rows.traversals, 2)

    def test_original_function_restored_on_exception(self):
        original = stability_report.verify_csv
        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            with linear_csv_verification():
                self.assertIs(stability_report.verify_csv, verify_csv)
                raise RuntimeError("interrupted")
        self.assertIs(stability_report.verify_csv, original)

    def test_portable_cpu_archive_has_identical_offline_result(self):
        archive = Path("experiments/runs/adamw_stability_20261002/bootstrap/mod193_cpu_smoke/portable")
        if not archive.exists():
            self.skipTest("Optional local full-width CPU smoke archive is unavailable")
        original = stability_report.verify_archive(archive)
        with linear_csv_verification():
            self.assertEqual(stability_report.verify_archive(archive), original)
        command = ["/usr/bin/python3", "-m",
                   "experiments.synthetic_trainers.protocols.adamw_stability_20261002.csv_verification",
                   "archive", str(archive)]
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(result.stdout),
                         {"verified": True, "name": original["name"], "torch_imported": False})


if __name__ == "__main__":
    unittest.main()
