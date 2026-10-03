"""The experiment files of the repository stay valid as the language changes."""

from pathlib import Path
import unittest

from lab.application.study import survey
from lab.infrastructure.loader import load

EXPERIMENTS = Path(__file__).resolve().parents[2] / "experiments"


def experiment_files():
    """The files under experiments/ written in the experiment language."""
    return sorted(path for path in EXPERIMENTS.glob("*.py") if "from lab.dsl import" in path.read_text())


class ExperimentFileTests(unittest.TestCase):
    def test_every_experiment_file_describes_its_experiments(self):
        files = experiment_files()
        self.assertTrue(files)
        for path in files:
            with self.subTest(path.name):
                rows = survey(load(path))
                self.assertEqual(len({row["fingerprint"] for row in rows}), len(rows))


if __name__ == "__main__":
    unittest.main()
