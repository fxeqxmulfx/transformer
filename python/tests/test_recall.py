"""The eager engine trains associative recall as the convex MQAR trainer did, bit for bit.

Golden records: the `runs` of `fixtures/legacy_recall.json`, short CPU runs of
`experiments/convex_mqar` `engine.train_run` in float32 at the commit the
fixture names: the hashes of every split, each epoch's validation metrics and
best epoch, the test metrics of the best model, the validation milestones
without their times, and the best and final parameters. The engine also
observes the initial model, at update 0, which the historical trainer did not.
"""

import math
from pathlib import Path
import tempfile
import unittest

from lab.domain.spec import swap
from lab.dsl import (AdamW, AssociativeRecall, Budget, Checkpoint, Eager, Evaluate, Experiment, Schedule, Seeds)
from lab.infrastructure.benchmarks import build_task
from lab.infrastructure.store import RunDirectory

from test_engine import Interrupted, sha, train
from test_nn import RECALL, recall_model, renamed


def recipe(config, length, width, rate, attention, batch):
    """`train_run` of a configuration in the DSL, at one length, width and rate: each epoch observed after its end."""
    epoch = math.ceil(config["train_examples"] / batch)
    updates = epoch * config["epochs"]
    return Experiment(
        model=recall_model(config, length, width, attention),
        benchmark=AssociativeRecall(length=length, vocab=config["vocab"], alpha=config["alpha"],
                                    train=config["train_examples"], validation=config["validation_examples"],
                                    test=config["test_examples"]),
        optimizer=AdamW(lr=rate, betas=(0.9, 0.999), weight_decay=config["weight_decay"], decay="matrices"),
        schedule=Schedule(warmup=max(1, int(config["warmup_fraction"] * updates)), inclusive=True),
        budget=Budget(updates=updates, batch=batch),
        seeds=Seeds(model=config["seed"], data=config["seed"], batches=config["seed"]),
        evaluate=Evaluate(every=epoch, batch=batch),
        execution=Eager(device="cpu"))


def experiment(name):
    """A fixture run in the DSL, with the batch it trained on."""
    golden = RECALL["runs"][name]
    config = golden["config"]
    return recipe(config, config["lengths"][0], config["widths"][0], config["learning_rates"][0],
                  golden["attention"], golden["batch"])


def metrics(golden):
    """Historical metrics under the current names."""
    return {"accuracy": golden["accuracy"], "loss": golden["loss"], "correct": golden["correct"],
            "queries": golden["query_count"]}


def milestone(golden, epoch):
    """A historical milestone at the observations of the current run, `epoch` updates apart."""
    if golden is None:
        return None
    # Each crossing in the fixture follows another epoch, never the initial model the historical trainer skipped.
    return {"step": golden["epoch"] * epoch, "accuracy": golden["validation_accuracy"],
            "previous_step": golden["previous_validation_epoch"] * epoch,
            "previous_accuracy": golden["previous_validation_accuracy"], "sustained_to_end": golden["sustained_to_end"]}


def parameters(state):
    """Hashes of a model state; the tied readout is the embedding, which a historical state holds once."""
    return {name: sha(value) for name, value in state.items() if name != "readout.weight"}


class HistoricalRecallTests(unittest.TestCase):
    def test_every_split_is_the_historical_split(self):
        for name, golden in RECALL["runs"].items():
            with self.subTest(run=name):
                benchmark, seed = experiment(name).benchmark, golden["config"]["seed"]
                task = build_task(benchmark, seed, "cpu")
                rows = {"train": task.train, **task.splits}
                for split, expected in golden["splits"].items():
                    tokens, positions, labels = rows[split].split([task.length, task.pairs, task.pairs], dim=1)
                    self.assertEqual({"tokens": sha(tokens), "positions": sha(positions), "labels": sha(labels)},
                                     expected)

    def assert_reproduces(self, name, root):
        golden = RECALL["runs"][name]
        epoch = experiment(name).evaluate.every
        run = RunDirectory(Path(root) / "run")
        history = run.records("history")
        self.assertEqual([(row["step"], row["epochs_seen"], row["validation"]) for row in history[1:]],
                         [(row["epoch"] * epoch, row["epoch"], metrics(row["validation"]))
                          for row in golden["history"]])
        result = run.result()
        self.assertEqual((result["best"]["step"], result["best"]["validation"], result["best"]["test"]),
                         (golden["best_epoch"] * epoch, metrics(golden["validation_best"]), metrics(golden["test"])))
        self.assertEqual(result["milestones"], {percent: milestone(hit, epoch)
                                                for percent, hit in golden["milestones"].items()})
        checkpoint = run.checkpoint("cpu")
        self.assertEqual(parameters(checkpoint["model"]), dict(renamed(golden["parameters"]["final"])))
        self.assertEqual(parameters(checkpoint["best"]["model"]), dict(renamed(golden["parameters"]["best"])))

    def test_every_historical_run(self):
        for name in RECALL["runs"]:
            with self.subTest(run=name), tempfile.TemporaryDirectory() as root:
                train(experiment(name), root)
                self.assert_reproduces(name, root)

    def test_an_interrupted_run_resumes_onto_the_historical_records(self):
        resumable = swap(experiment("softmax"), "checkpoint", Checkpoint(every=8))

        def interrupt(label, row):
            if row["step"] == 12:
                raise Interrupted

        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(Interrupted):
                train(resumable, root, interrupt)
            self.assertEqual(RunDirectory(Path(root) / "run").checkpoint("cpu")["step"], 8)
            train(resumable, root)
            self.assert_reproduces("softmax", root)

    def test_a_run_resumed_within_an_epoch_draws_the_rest_of_its_batches(self):
        # Three updates an epoch; the checkpoint after update 4 lies one batch into the second epoch.
        resumable = swap(swap(experiment("heads"), "evaluate.every", 2), "checkpoint", Checkpoint(every=4))

        def interrupt(label, row):
            if row["step"] == 6:
                raise Interrupted

        with tempfile.TemporaryDirectory() as root:
            straight, resumed = Path(root) / "straight", Path(root) / "resumed"
            train(resumable, straight)
            with self.assertRaises(Interrupted):
                train(resumable, resumed, interrupt)
            checkpoint = RunDirectory(resumed / "run").checkpoint("cpu")
            self.assertEqual((checkpoint["step"], checkpoint["epoch"], checkpoint["cursor"]), (4, 1, 64))
            train(resumable, resumed)
            records = [RunDirectory(path / "run").records("history") for path in (straight, resumed)]
            self.assertEqual(*[[{key: value for key, value in row.items() if key not in ("training_seconds",
                                                                                       "wall_seconds")}
                                for row in rows] for rows in records])
            self.assertEqual(*[parameters(RunDirectory(path / "run").checkpoint("cpu")["model"])
                               for path in (straight, resumed)])

    def test_a_sequence_holds_distinct_keys_and_their_queries(self):
        recall = experiment("softmax").benchmark
        with self.assertRaisesRegex(ValueError, "at least one key-value pair"):
            swap(recall, "length", 3)
        with self.assertRaisesRegex(ValueError, "Distinct keys"):
            swap(recall, "vocab", 6)


if __name__ == "__main__":
    unittest.main()
