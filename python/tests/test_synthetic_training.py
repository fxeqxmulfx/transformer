"""The eager engine trains a synthetic benchmark as the historical synthetic trainer did, bit for bit.

Golden records: `fixtures/legacy_synthetic.json`, short CPU runs of
`experiments/synthetic_trainers/training.train_run` without a study, at the
commit the fixture names: every observation's validation metrics, the best
observation and the test metrics of its model, the time to target, the split
fingerprints, and the best and final parameters.
"""

import json
from pathlib import Path
import tempfile
import unittest

import torch

from lab.domain.model import Normal, TorchDefault
from lab.domain.spec import swap
from lab.dsl import (AdamW, AMSGradW, Budget, Checkpoint, Clipped, Eager, Evaluate, Experiment, Schedule, Seeds,
                     Synthetic)
from lab.infrastructure.benchmarks import build_task
from lab.infrastructure.benchmarks.synthetic import generator
from lab.infrastructure.benchmarks.synthetic.rows import Rows
from lab.infrastructure.benchmarks.synthetic.splits import build_split
from lab.infrastructure.nn.legacy import rename
from lab.infrastructure.store import RunDirectory

from examples import gptmini
from test_engine import Interrupted, sha, train
from test_synthetic import block

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "legacy_synthetic.json").read_text())


def current(name):
    """A historical split's current name."""
    if name in ("train", "validation"):
        return name
    return "test" if name == "in_distribution" else f"test/{name}"


def experiment(golden):
    """A fixture run in the DSL."""
    task, model, training = golden["task"], golden["model"], golden["training"]
    benchmark = Synthetic(task=block(task), length=task["length"], min_length=task["min_length"],
                          train=training["train_examples"], validation=training["validation_examples"],
                          test=training["test_examples"], ood=tuple(training["eval_lengths"]),
                          target=training["target"], metric=training["target_metric"])
    init = TorchDefault() if model["init_std"] is None else Normal(model["init_std"])
    architecture = swap(swap(gptmini(model["width"], model["layers"], model["heads"]), "context", benchmark.context),
                        "init", init)
    betas = (training["beta1"], training["beta2"])
    if training["optimizer"] == "adamw":
        rule = AdamW(lr=training["learning_rate"], betas=betas, weight_decay=training["weight_decay"],
                     eps=training["optimizer_epsilon"], decay="matrices")
    else:
        rule = AMSGradW(lr=training["learning_rate"], betas=betas, weight_decay=training["weight_decay"],
                        eps=training["optimizer_epsilon"])
    if training["grad_clip"] is not None:
        rule = Clipped(rule, norm=training["grad_clip"])
    return Experiment(model=architecture, benchmark=benchmark, optimizer=rule, schedule=Schedule(),
                      budget=Budget(updates=training["steps"], batch=training["batch_size"]),
                      seeds=Seeds(model=training["seed"], data=training["data_seed"], batches=training["seed"]),
                      evaluate=Evaluate(every=training["eval_every"], batch=training["batch_size"]),
                      execution=Eager(device="cpu", threads=training["cpu_threads"]))


def renamed(golden):
    """The digest of historical parameters under the current names."""
    return {rename(name): value for name, value in golden.items()}


def untimed(hit):
    return None if hit is None else (hit["step"], hit["examples_seen"], hit["value"])


class HistoricalTrainingTests(unittest.TestCase):
    def assert_reproduces(self, golden, root):
        run = RunDirectory(Path(root) / "historical" / "run")
        self.assertEqual([{key: row[key] for key in ("step", "examples_seen", "validation")}
                          for row in run.records("history")], golden["history"])
        result = run.result()
        best = result["best"]
        self.assertEqual((best["step"], best["validation"]), (golden["best_step"], golden["validation_best"]))
        self.assertEqual({name: best[current(name)] for name in golden["test"]}, golden["test"])
        self.assertEqual(untimed(result["time_to_target"]), untimed(golden["time_to_target"]))
        self.assertEqual(result["split_fingerprints"],
                         {current(name): value for name, value in golden["split_fingerprints"].items()})
        checkpoint = run.checkpoint("cpu")
        self.assertEqual({key: sha(value) for key, value in checkpoint["model"].items()},
                         renamed(golden["parameters"]["final"]))
        self.assertEqual({key: sha(value) for key, value in checkpoint["best"]["model"].items()},
                         renamed(golden["parameters"]["best"]))

    def test_every_historical_run(self):
        for name, golden in FIXTURE["runs"].items():
            with self.subTest(run=name), tempfile.TemporaryDirectory() as root:
                train(experiment(golden), root)
                self.assert_reproduces(golden, root)

    def test_an_interrupted_run_resumes_onto_the_same_records(self):
        # Its best observation, at update 20, travels in the checkpoint the run resumes from.
        golden = FIXTURE["runs"]["crasp-amsgradw-clipped"]
        resumable = swap(experiment(golden), "checkpoint", Checkpoint(every=10))

        def interrupt(label, row):
            if row["step"] == 30:
                raise Interrupted

        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(Interrupted):
                train(resumable, root, interrupt)
            self.assertEqual(RunDirectory(Path(root) / "historical" / "run").checkpoint("cpu")["best"]["step"], 20)
            train(resumable, root)
            self.assert_reproduces(golden, root)

    def test_the_model_reads_the_historical_vocabulary_and_context(self):
        for name, golden in FIXTURE["runs"].items():
            with self.subTest(run=name):
                benchmark = experiment(golden).benchmark
                task = build_task(benchmark, golden["training"]["data_seed"], "cpu")
                self.assertEqual((task.vocab, benchmark.context), (golden["vocab_size"], golden["context_length"]))


class RowsTests(unittest.TestCase):
    def test_a_chunk_is_cut_to_its_longest_row_and_keeps_its_supervision(self):
        benchmark = experiment(FIXTURE["runs"]["dyck-amsgradw"]).benchmark
        split = build_split(generator(benchmark.problem), "validation", 1, 16)
        rows = Rows(split.examples, "cpu")
        chunk = rows[3:9]
        self.assertIs(rows[3:9], chunk)
        self.assertEqual(chunk.rows.shape[-1], max(len(example.tokens) for example in split.examples[3:9]))
        for row, example in zip(chunk.rows, split.examples[3:9]):
            size = len(example.tokens)
            self.assertEqual([tuple(field[:size].tolist()) for field in row],
                             [example.tokens, example.targets, tuple(map(int, example.changes))])
        supervised = chunk.rows[:, 1].flatten()[chunk.supervised]
        self.assertEqual(supervised.tolist(), [target for example in split.examples[3:9]
                                               for target in example.targets if target != -100])
        self.assertEqual(rows.select(torch.tensor([0, 5])).shape[-1],
                         max(len(split.examples[index].tokens) for index in (0, 5)))


if __name__ == "__main__":
    unittest.main()
