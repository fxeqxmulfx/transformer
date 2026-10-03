"""The eager engine reproduces the historical modular trainer, record for record.

Golden records: the `runs` and `cuda_runs` of `fixtures/legacy_modular.json`,
written by `paper_reproduction.grokking.train` under AdamW and AMSGradW, and by
its sparsemax and cosine variants, at the commit the fixture names. Timing
fields are left out.
"""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import torch

from lab.application.study import Study, run_study
from lab.domain.model import Softmax, Sparsemax
from lab.domain.spec import substitute, swap
from lab.domain.optimizers import AMSGradW
from lab.domain.training import Checkpoint, Cosine, Diagnostics, Eager
from lab.infrastructure.engine import Engine
from lab.infrastructure.nn.legacy import DERIVED, rename
from lab.infrastructure.store import RunDirectories, RunDirectory

from examples import gptmini, modular, reference

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "legacy_modular.json").read_text())
TIMING = {"training_seconds", "wall_seconds"}
TRACE = Diagnostics(every=4, neighbors=True, gradients=True)


def historical(name, device="cpu"):
    """A fixture run in the DSL: prime 11, 30 updates of 8 rows, observed every 10."""
    model = reference(32, 2, 4) if name.startswith("reference") else gptmini(32, 2, 4)
    if name == "gptmini-sparsemax":
        model = substitute(model, Softmax, Sparsemax())
    experiment = modular(model, prime=11, updates=30, batch=8, every=10, execution=Eager(device=device))
    if name == "gptmini-wrap":
        return swap(experiment, "benchmark.tail", "wrap")
    if name in ("gptmini-cosine", "reference-amsgradw"):
        experiment = swap(experiment, "diagnostics", Diagnostics(gradients=True))
    else:
        experiment = swap(experiment, "diagnostics", TRACE)
    if name == "gptmini-cosine":
        return swap(experiment, "schedule.anneal", Cosine(start=15, end=25, final=.1))
    if name.endswith("-amsgradw"):
        return swap(experiment, "optimizer", AMSGradW(lr=1e-3, betas=(0.9, 0.999), weight_decay=1.0))
    return experiment


def train(experiment, root, progress=lambda label, row: None):
    study = Study("historical", "# a test study\n", {"run": experiment})
    return run_study(study, [], RunDirectories(root), Engine(), progress)["run"]


def sha(tensor):
    return hashlib.sha256(tensor.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def digest(checkpoint):
    """A checkpoint as the fixture summarizes it."""
    return {"step": checkpoint["step"], "examples_seen": checkpoint["examples_seen"],
            "last_batch_size": checkpoint["last_batch_size"], "cursor": checkpoint["cursor"],
            "permutation": checkpoint["permutation"].tolist(), "generator": sha(checkpoint["batch_generator_state"]),
            "model": [(name, sha(value)) for name, value in checkpoint["model"].items()],
            "optimizer": {str(index): {key: sha(value) if value.ndim else float(value) for key, value in entry.items()}
                          for index, entry in checkpoint["optimizer"]["state"].items()}}


def current_names(golden):
    """Fixture records under the current parameter names, in the historical order."""
    if "model" in golden:
        return {**golden, "model": [(rename(name), value) for name, value in golden["model"].items()
                                    if name not in DERIVED]}
    return {**golden, "parameters": {rename(name): value for name, value in golden["parameters"].items()},
            "temperatures": {rename(name): value for name, value in golden["temperatures"].items()}}


def untimed(rows):
    return [{key: value for key, value in row.items() if key not in TIMING} for row in rows]


class Interrupted(Exception):
    pass


class HistoricalRunTests(unittest.TestCase):
    def assert_reproduces(self, golden, root):
        run = RunDirectory(Path(root) / "run")
        self.assertEqual(untimed(run.records("history")), golden["history"])
        self.assertEqual(untimed(run.records("probes")), golden["probes"])
        self.assertEqual(run.records("gradients"), golden["gradients"])
        sampled = run.records("diagnostics")
        self.assertEqual(sampled, [current_names(row) for row in golden["sampled"]])
        self.assertEqual([list(row["parameters"]) for row in sampled],
                         [list(current_names(row)["parameters"]) for row in golden["sampled"]])
        self.assertEqual(run.result()["transition"], golden["transition"])
        self.assertEqual(digest(run.checkpoint("cpu")), current_names(golden["checkpoint"]))

    def test_every_historical_cpu_run(self):
        for name, golden in FIXTURE["runs"].items():
            with self.subTest(run=name), tempfile.TemporaryDirectory() as root:
                train(historical(name), root)
                self.assert_reproduces(golden, root)

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_every_historical_cuda_run_on_the_fixture_gpu(self):
        if FIXTURE["source"].get("gpu") != torch.cuda.get_device_name():
            self.skipTest("CUDA golden records were written on another GPU")
        for name, golden in FIXTURE["cuda_runs"].items():
            with self.subTest(run=name), tempfile.TemporaryDirectory() as root:
                train(historical(name, device="cuda"), root)
                self.assert_reproduces(golden, root)

    def test_an_interrupted_run_resumes_onto_the_same_records(self):
        experiment = swap(historical("gptmini"), "checkpoint", Checkpoint(every=10))

        def interrupt(label, row):
            if row["step"] == 21:
                raise Interrupted

        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(Interrupted):
                train(experiment, root, interrupt)
            self.assertEqual(RunDirectory(Path(root) / "run").checkpoint("cpu")["step"], 20)
            train(experiment, root)
            self.assert_reproduces(FIXTURE["runs"]["gptmini"], root)

    def test_an_extended_budget_continues_the_trajectory(self):
        experiment = historical("gptmini")
        with tempfile.TemporaryDirectory() as root:
            first = train(swap(experiment, "budget.updates", 20), root)
            self.assertEqual(first["final"]["step"], 20)
            train(experiment, root)
            self.assert_reproduces(FIXTURE["runs"]["gptmini"], root)
            run = RunDirectory(Path(root) / "run")
            self.assertEqual([segment["updates"] for segment in run.manifest()["segments"]], [20, 30])
            self.assertEqual(train(experiment, root), run.result())
            self.assertEqual(len(run.manifest()["segments"]), 2)


if __name__ == "__main__":
    unittest.main()
