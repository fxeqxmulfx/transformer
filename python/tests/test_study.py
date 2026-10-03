"""The study use case: which runs train, and what may continue an existing run."""

import unittest

from lab.application.study import Study, run_study
from lab.domain.spec import swap

from examples import gptmini, modular


class FakeRun:
    def __init__(self):
        self.stored, self.engine, self.summary = None, None, None

    def description(self):
        return self.stored

    def provenance(self):
        return {"engine": self.engine}

    def result(self):
        return self.summary

    def begin(self, description, provenance, source):
        self.stored, self.engine, self.summary = description, provenance["engine"], None


class FakeRuns:
    def __init__(self):
        self.runs = {}

    def open(self, study, label):
        return self.runs.setdefault((study, label), FakeRun())


class FakeTrainer:
    def __init__(self, engine="v1"):
        self.engine, self.trained = engine, []

    def provenance(self, experiment):
        return {"engine": self.engine}

    def train(self, experiment, run, progress):
        self.trained.append(experiment.budget.updates)
        run.summary = {"updates": experiment.budget.updates}
        return run.summary


def study(**experiments):
    return Study("pair", "# source\n", experiments)


class RunStudyTests(unittest.TestCase):
    def setUp(self):
        self.base = modular(gptmini(), updates=100)
        self.runs = FakeRuns()

    def test_a_finished_run_is_not_trained_again(self):
        trainer = FakeTrainer()
        run_study(study(base=self.base), [], self.runs, trainer, print)
        self.assertEqual(run_study(study(base=self.base), [], self.runs, trainer, print),
                         {"base": {"updates": 100}})
        self.assertEqual(trainer.trained, [100])

    def test_an_extended_budget_trains_again(self):
        trainer = FakeTrainer()
        run_study(study(base=self.base), [], self.runs, trainer, print)
        run_study(study(base=swap(self.base, "budget.updates", 200)), [], self.runs, trainer, print)
        self.assertEqual(trainer.trained, [100, 200])

    def test_a_changed_experiment_stops_the_study_before_any_update(self):
        run_study(study(other=self.base), [], self.runs, FakeTrainer(), print)
        trainer = FakeTrainer()
        changed = study(base=self.base, other=swap(self.base, "optimizer.lr", 3e-4))
        with self.assertRaisesRegex(ValueError, "changed: optimizer.lr"):
            run_study(changed, [], self.runs, trainer, print)
        self.assertEqual(trainer.trained, [])

    def test_another_engine_continues_no_unfinished_run(self):
        run_study(study(base=self.base), [], self.runs, FakeTrainer("v1"), print)
        extended = study(base=swap(self.base, "budget.updates", 200))
        trainer = FakeTrainer("v2")
        with self.assertRaisesRegex(ValueError, "another engine"):
            run_study(extended, [], self.runs, trainer, print)
        self.assertEqual(run_study(study(base=self.base), [], self.runs, trainer, print),
                         {"base": {"updates": 100}})
        self.assertEqual(trainer.trained, [])


if __name__ == "__main__":
    unittest.main()
