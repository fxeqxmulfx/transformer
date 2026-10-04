"""The basis runs a model on depth, recall and parity; the hard mode asks more depth and recall, and the same parity."""

import unittest

from examples import gptmini
from lab.dsl import Schedule, Solved, basis, swap

MODES = ("easy", "hard")


class BasisTests(unittest.TestCase):
    def test_each_mode_runs_the_model_on_every_task_until_solved(self):
        model = gptmini(64, 2)
        for mode in MODES:
            runs = basis(model, mode)
            self.assertEqual(list(runs), ["depth", "recall", "parity"])
            for task, run in runs.items():
                with self.subTest(mode=mode, task=task):
                    self.assertEqual(run.model, swap(model, "context", run.benchmark.context))
                    self.assertEqual((run.benchmark.target, run.benchmark.metric), (0.99, "sequence_accuracy"))
                    self.assertEqual(run.stopping, Solved())
                    self.assertEqual(run.evaluate.every * run.budget.batch, 3_200)

    def test_the_hard_mode_is_harder_on_depth_and_recall_and_keeps_parity(self):
        easy, hard = (basis(gptmini(), mode) for mode in MODES)
        self.assertEqual((easy["depth"].benchmark.task.blocks, hard["depth"].benchmark.task.blocks), (2, 4))
        self.assertEqual((easy["recall"].benchmark.task.overwrites, hard["recall"].benchmark.task.overwrites), (0, 8))
        self.assertLess(easy["recall"].benchmark.task.pairs, hard["recall"].benchmark.task.pairs)
        self.assertEqual(easy["parity"], hard["parity"])

    def test_hard_depth_is_judged_beyond_its_training_lengths_and_easy_depth_only_tested_there(self):
        easy, hard = (basis(gptmini(), mode)["depth"].benchmark for mode in MODES)
        self.assertEqual((easy.selection, hard.selection), ("validation", "validation/length-128"))
        for benchmark in (easy, hard):
            self.assertEqual(benchmark.ood, (128,))
            self.assertLess(benchmark.length, 128)

    def test_every_run_holds_its_rate_after_a_warmup_of_50_updates(self):
        for mode in MODES:
            for task, run in basis(gptmini(), mode).items():
                with self.subTest(mode=mode, task=task):
                    self.assertEqual(run.schedule, Schedule(warmup=50))

    def test_a_seed_moves_the_model_and_its_batches_alone(self):
        first, second = basis(gptmini(), "easy"), basis(gptmini(), "easy", seed=1)
        for task, run in first.items():
            with self.subTest(task=task):
                self.assertEqual(swap(second[task], "seeds", run.seeds), run)
                self.assertEqual(second[task].seeds.data, run.seeds.data)
                self.assertNotEqual((second[task].seeds.model, second[task].seeds.batch_seed),
                                    (run.seeds.model, run.seeds.batch_seed))

    def test_an_unknown_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            basis(gptmini(), "medium")


if __name__ == "__main__":
    unittest.main()
