"""The experiments of the repository stay valid as the language changes.

An experiment written from archived runs trains their recipes:
`synthetic_amsgradw` those of `fixtures/legacy_baseline.json`, the manifest of
the archived AMSGradW/softmax baseline of the synthetic suite,
`synthetic_scaling` those of `fixtures/legacy_scaling.json`, the
configurations its archived sweeps recorded, and `mqar_sparsemax` the `full`
and `sanity` profiles of the convex MQAR comparison recorded in
`fixtures/legacy_recall.json`, on the lab's MQAR.
"""

import itertools
import json
import math
from pathlib import Path
import unittest

from lab.application.study import survey
from lab.domain.spec import swap
from lab.dsl import (MQAR, AdamW, Budget, Checkpoint, CudaGraph, Eager, Evaluate, Experiment, Schedule, Seeds,
                     Synthetic)
from lab.infrastructure.loader import load

from test_memorization_training import study
from test_nn import RECALL, recall_model

EXPERIMENTS = Path(__file__).resolve().parents[2] / "experiments"
FIXTURES = Path(__file__).parent / "fixtures"


def experiment_folders():
    """The experiments under experiments/: the folders holding an `experiment.py`."""
    return sorted(path.parent for path in EXPERIMENTS.glob("*/experiment.py"))


class ExperimentFolderTests(unittest.TestCase):
    def test_every_experiment_describes_its_runs(self):
        folders = experiment_folders()
        self.assertTrue(folders)
        for path in folders:
            with self.subTest(path.name):
                rows = survey(load(path))
                self.assertEqual(len({row["fingerprint"] for row in rows}), len(rows))

    def test_every_experiment_has_a_readme(self):
        for path in experiment_folders():
            with self.subTest(path.name):
                self.assertTrue((path / "README.md").is_file())


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


# Each experiment written from archived runs, the fixture of their recipes, and the label it gives a run.
ARCHIVED = {"synthetic_amsgradw": ("legacy_baseline.json", baseline),
            "synthetic_scaling": ("legacy_scaling.json", scaling)}


def recipe(config, length, width, rate, attention, batch):
    """`train_run` of a configuration in the DSL, at one length, width and rate: each epoch observed after its end.

    The comparison's sequences of `length` tokens bound `length / 4` keys out
    of half its vocabulary to values out of the other half, queried each
    once at positions weighted p^-alpha; here the lab's MQAR draws them, and
    its target is the comparison's milestone, 99% of the queries answered.
    """
    epoch = math.ceil(config["train_examples"] / batch)
    updates = epoch * config["epochs"]
    pairs = length // 4
    return Experiment(
        model=recall_model(config, length, width, attention),
        benchmark=Synthetic(MQAR(symbols=config["vocab"] // 2, pairs=pairs, queries=pairs, alpha=config["alpha"]),
                            length=length, train=config["train_examples"], validation=config["validation_examples"],
                            test=config["test_examples"], target=.99, metric="token_accuracy"),
        optimizer=AdamW(lr=rate, betas=(0.9, 0.999), weight_decay=config["weight_decay"], decay="matrices"),
        schedule=Schedule(warmup=max(1, int(config["warmup_fraction"] * updates)), inclusive=True),
        budget=Budget(updates=updates, batch=batch),
        seeds=Seeds(model=config["seed"], data=config["seed"], batches=config["seed"]),
        evaluate=Evaluate(every=epoch, batch=batch),
        execution=Eager(device="cpu"))


def recall():
    """The runs of the archived MQAR profiles: `full` under either attention, at every length and rate, and `sanity`."""
    runs = {}
    for profile, attentions in (("full", ("softmax", "sparsemax")), ("sanity", ("softmax",))):
        config, batches = RECALL["profiles"][profile]["config"], RECALL["profiles"][profile]["batches"]
        for attention, row, rate in itertools.product(attentions, batches, config["learning_rates"]):
            run = recipe(config, row["length"], row["width"], rate, attention, row["batch"])
            label = profile if profile == "sanity" else f"{attention}-n{row['length']}-lr{rate:.8g}"
            runs[label] = swap(swap(run, "execution", CudaGraph()), "checkpoint", Checkpoint(every=run.evaluate.every))
    return runs


class ArchivedRecipeTests(unittest.TestCase):
    def test_experiments_written_from_archived_runs_train_their_recipes(self):
        for name, (fixture, label) in ARCHIVED.items():
            with self.subTest(name):
                recipes = json.loads((FIXTURES / fixture).read_text())["recipes"]
                self.assertEqual(load(EXPERIMENTS / name).experiments,
                                 {label(recipe["name"]): swap(study(recipe), "execution", CudaGraph())
                                  for recipe in recipes})

    def test_the_recall_experiment_trains_the_archived_profiles(self):
        self.assertEqual(load(EXPERIMENTS / "mqar_sparsemax").experiments, recall())


if __name__ == "__main__":
    unittest.main()
