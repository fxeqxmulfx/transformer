"""The baseline matrix has valid OOD controls and separated transition pools."""

from dataclasses import replace
import unittest

from experiments.synthetic_trainers.baseline_recipes import recipes
from experiments.synthetic_trainers.corpus import corpus_report, study_pool
from experiments.synthetic_trainers.data import build_split


class BaselineRecipesTests(unittest.TestCase):
    def test_complete_plan_has_unique_runs_and_valid_oracles_for_every_probe(self):
        plan = recipes(device="cpu")
        self.assertEqual(len(plan), 61)
        self.assertEqual(len({run.name for run in plan}), len(plan))
        suite = [run for run in plan if run.phase == "suite"]
        self.assertEqual(len(suite), 37)
        self.assertEqual(len({run.task.task for run in suite}), 17)
        for run in plan:
            self.assertEqual(run.training.optimizer, "amsgradw")
            self.assertIsNone(run.training.grad_clip)
            self.assertEqual(run.model.init_std, .02)
            build_split(run.task, "train", 1, 4)
            for length in run.training.eval_lengths:
                build_split(replace(run.task, length=length, min_length=None), "validation", 1, 4)

    def test_fixed_parity_domain_can_supply_every_disjoint_split(self):
        run = next(run for run in recipes("transitions", device="cpu", seeds=(0,)) if run.task.task == "parity")
        report = corpus_report(study_pool(run.task, run.training))
        self.assertEqual(sum(row["unique_inputs"] for row in report["splits"].values()), 256)
        self.assertFalse(any(row["unique_inputs"] for row in report["overlaps"].values()))

    def test_model_seed_changes_do_not_change_pools(self):
        first, second = recipes("transitions", device="cpu", seeds=(0, 1))[:2]
        pool_a, pool_b = study_pool(first.task, first.training), study_pool(second.task, second.training)
        self.assertEqual({name: row.fingerprint for name, row in pool_a.items()},
                         {name: row.fingerprint for name, row in pool_b.items()})
