"""The experiment files of the repository stay valid as the language changes.

A file written from archived runs trains their recipes: `synthetic_amsgradw.py`
those of `fixtures/legacy_baseline.json`, the manifest of the archived
AMSGradW/softmax baseline of the synthetic suite, and `synthetic_scaling.py`
those of `fixtures/legacy_scaling.json`, the configurations its archived
sweeps recorded.
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
FIXTURES = Path(__file__).parent / "fixtures"


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


def baseline(name):
    """The label of a run of the archived baseline, `phase/variant/width-W/noise-r/seed-s`."""
    phase, variant, width, _, seed = name.split("/")
    seed = seed.replace("-", "")
    return {"suite": f"suite-{variant.removesuffix('-program-0')}", "transitions": f"transitions-{variant}-{seed}",
            "capacity": f"capacity-{width.replace('-', '')}-{seed}", "control": f"control-{seed}"}[phase]


def scaling(name):
    """The label of a run of the archived sweeps, `phase/width-W-layers-L/samples-N/noise-r/data-d-seed-s`."""
    phase, model, samples, _, seeds = name.split("/")
    width, layers = model.split("-")[1::2]
    seed = seeds.split("-")[-1]
    return {"dd-calibration": f"calibration-{samples.replace('-', '')}",
            "dd-widths": f"widths-width{width}-seed{seed}", "copy-scaling": f"copy-width{width}-depth{layers}"}[phase]


# Each file written from archived runs, the fixture of their recipes, and the label the file gives a run.
ARCHIVED = {"synthetic_amsgradw.py": ("legacy_baseline.json", baseline),
            "synthetic_scaling.py": ("legacy_scaling.json", scaling)}


class ArchivedRecipeTests(unittest.TestCase):
    def test_files_written_from_archived_runs_train_their_recipes(self):
        for name, (fixture, label) in ARCHIVED.items():
            with self.subTest(name):
                recipes = json.loads((FIXTURES / fixture).read_text())["recipes"]
                self.assertEqual(load(EXPERIMENTS / name).experiments,
                                 {label(recipe["name"]): swap(study(recipe), "execution", CudaGraph())
                                  for recipe in recipes})


if __name__ == "__main__":
    unittest.main()
