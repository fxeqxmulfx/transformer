"""The eager engine trains a memorization study as the historical synthetic trainer did, bit for bit.

Golden records: `fixtures/legacy_memorization.json`, short CPU runs of
`experiments/synthetic_trainers/training.train_run` under its memorization and
double-descent studies, at the commit the fixture names: each observation's
metrics of every observed split, with the compression of the random control
and the fit of noisy labels; the best observation, and the test metrics of
its model and of the last; the study's report, and its diagnostics of both
models; the split fingerprints; and the best and final parameters. Free
generation is recorded without its cost counters and time, which the port
does not report.
"""

import json
from pathlib import Path
import tempfile
import unittest

from lab.domain.memorization import Memorization
from lab.domain.spec import swap
from lab.dsl import Checkpoint
from lab.infrastructure.benchmarks import build_task
from lab.infrastructure.store import RunDirectory

from test_engine import Interrupted, sha, train
from test_synthetic_training import current, experiment, renamed, untimed

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "legacy_memorization.json").read_text())
# Times, and the size of the last batch, which the historical history did not record.
UNCOMPARED = {"training_seconds", "wall_seconds", "last_batch_size"}
# The labels of the historical report, and what the port reports elsewhere: the diagnostics of the last and the
# best model beside their test metrics, and the held-out validation fingerprints beside the other splits'.
RELOCATED = {"profile", "data_scope", "final_checkpoint", "best_checkpoint", "validation_probe_fingerprints"}


def study(golden):
    """A fixture run and its study in the DSL."""
    training = golden["training"]
    return swap(experiment(golden), "benchmark.study", Memorization(
        disjoint=training["split_policy"] == "disjoint", noise=training["label_noise"],
        noise_seed=training["noise_seed"], fit=training["fit_metric"], epsilon=training["fit_epsilon"],
        patience=training["generalization_patience"], tolerance=training["curve_tolerance"]))


def observation(golden):
    """A historical observation under the current split names."""
    row = {key: golden[key] for key in ("step", "examples_seen", "epochs_seen", "train", "train_clean", "validation")}
    row["validation/novel"] = golden["validation_novel"]
    for name, metrics in golden["validation_ood"].items():
        row[f"validation/{name}"], row[f"validation/{name}/novel"] = metrics, golden["validation_ood_novel"][name]
    return {**row, **{key: golden[key] for key in ("compression", "noise_fit") if key in golden}}


def fingerprints(golden):
    """The historical split fingerprints under the current names."""
    found = {name if name == "train_clean" else current(name): value
             for name, value in golden["split_fingerprints"].items()}
    return {**found, **{f"validation/{name}": value
                        for name, value in golden["study"]["validation_probe_fingerprints"].items()}}


class HistoricalStudyTests(unittest.TestCase):
    def assert_reproduces(self, golden, root):
        run = RunDirectory(Path(root) / "run")
        self.assertEqual([{key: value for key, value in row.items() if key not in UNCOMPARED}
                          for row in run.records("history")], [observation(row) for row in golden["history"]])
        result = run.result()
        best, last = result["best"], result["last"]
        self.assertEqual((best["step"], best["validation"]), (golden["best_step"], golden["validation_best"]))
        self.assertEqual({name: best[current(name)] for name in golden["test"]}, golden["test"])
        self.assertEqual({name: last[current(name)] for name in golden["test_final"]}, golden["test_final"])
        report = golden["study"]
        self.assertEqual(result["memorization"], {key: value for key, value in report.items() if key not in RELOCATED})
        self.assertEqual((last["memorization"], best["memorization"]),
                         (report["final_checkpoint"], report["best_checkpoint"]))
        self.assertEqual(untimed(result["time_to_target"]), untimed(golden["time_to_target"]))
        self.assertEqual(result["split_fingerprints"], fingerprints(golden))
        checkpoint = run.checkpoint("cpu")
        self.assertEqual({key: sha(value) for key, value in checkpoint["model"].items()},
                         renamed(golden["parameters"]["final"]))
        self.assertEqual({key: sha(value) for key, value in checkpoint["best"]["model"].items()},
                         renamed(golden["parameters"]["best"]))

    def test_every_historical_study(self):
        for name, golden in FIXTURE["runs"].items():
            with self.subTest(run=name), tempfile.TemporaryDirectory() as root:
                benchmark = study(golden).benchmark
                task = build_task(benchmark, golden["training"]["data_seed"], "cpu")
                self.assertEqual((task.vocab, benchmark.context), (golden["vocab_size"], golden["context_length"]))
                train(study(golden), root)
                self.assert_reproduces(golden, root)

    def test_an_interrupted_study_resumes_onto_the_same_records(self):
        # Its best observation, at update 20, travels in the checkpoint the run resumes from.
        golden = FIXTURE["runs"]["parity-running-noise-adamw-clipped"]
        resumable = swap(study(golden), "checkpoint", Checkpoint(every=10))

        def interrupt(label, row):
            if row["step"] == 30:
                raise Interrupted

        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(Interrupted):
                train(resumable, root, interrupt)
            self.assertEqual(RunDirectory(Path(root) / "run").checkpoint("cpu")["best"]["step"], 20)
            train(resumable, root)
            self.assert_reproduces(golden, root)


if __name__ == "__main__":
    unittest.main()
