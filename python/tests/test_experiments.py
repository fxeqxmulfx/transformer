"""The experiment files of the repository stay valid as the language changes.

A file written from archived runs trains their recipes: `synthetic_amsgradw.py`
those of `fixtures/legacy_baseline.json`, the manifest of the archived
AMSGradW/softmax baseline of the synthetic suite.
"""

import json
from pathlib import Path
import unittest

from lab.application.study import survey
from lab.domain.spec import swap
from lab.domain.training import CudaGraph
from lab.infrastructure.loader import load

from test_memorization_training import study

EXPERIMENTS = Path(__file__).resolve().parents[2] / "experiments"
BASELINE = json.loads((Path(__file__).parent / "fixtures" / "legacy_baseline.json").read_text())


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


def label(name):
    """The label of an archived baseline run, `phase/variant/width-W/noise-r/seed-s`, in its experiment file."""
    phase, variant, width, _, seed = name.split("/")
    seed = seed.replace("-", "")
    return {"suite": f"suite-{variant.removesuffix('-program-0')}", "transitions": f"transitions-{variant}-{seed}",
            "capacity": f"capacity-{width.replace('-', '')}-{seed}", "control": f"control-{seed}"}[phase]


class ArchivedRecipeTests(unittest.TestCase):
    def test_the_synthetic_baseline_trains_its_archived_recipes(self):
        recipes = {label(recipe["name"]): swap(study(recipe), "execution", CudaGraph())
                   for recipe in BASELINE["recipes"]}
        self.assertEqual(load(EXPERIMENTS / "synthetic_amsgradw.py").experiments, recipes)


if __name__ == "__main__":
    unittest.main()
