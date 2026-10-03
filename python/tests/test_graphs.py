"""CUDA graph replays equal the same operations issued eagerly, bit for bit.

The reference is `GraphStepper` without `prepare`: nothing is captured, so
every update and evaluation is issued eagerly with the capturable optimizer.
"""

import json
from pathlib import Path
import tempfile
import unittest

import torch

from lab.domain.model import Softmax, Sparsemax
from lab.domain.optimizers import (SGD, AdaFisher, AdamNC, AdamW, AdamX, AMSGradMD, AMSGradW, Chebyshev, Clipped,
                                   Dash, Guarded, Magma, Muon)
from lab.domain.spec import describe, substitute, swap
from lab.domain.training import Checkpoint, Cosine, CudaGraph, Diagnostics, Evaluate
from lab.infrastructure.engine.graphs import GraphStepper
from lab.infrastructure.engine.loop import Training
from lab.infrastructure.store import STREAMS, RunDirectory

from examples import gptmini, modular, reference
from test_engine import TRACE, Interrupted, sha, untimed
from test_synthetic_training import FIXTURE as SYNTHETIC
from test_synthetic_training import experiment as synthetic
from test_text import FIXTURE as TEXT
from test_text import historical as text


class Issued(GraphStepper):
    """Every operation issued eagerly."""

    def prepare(self, sizes):
        pass


class Counted(GraphStepper):
    """Replays as usual, counting the updates issued eagerly instead."""

    def __init__(self, *arguments):
        super().__init__(*arguments)
        self.issued = 0

    def issue(self, parts, sampled):
        self.issued += 1
        return super().issue(parts, sampled)


def experiments():
    base = modular(gptmini(32, 2, 4), prime=11, updates=30, batch=8, every=10, execution=CudaGraph())
    gradients = swap(base, "diagnostics", Diagnostics(gradients=True))
    sparse = substitute(swap(gradients, "schedule.anneal", Cosine(start=15, end=25, final=.1)), Softmax, Sparsemax())
    return {"replayed": gradients, "sampled": swap(base, "diagnostics", TRACE),
            "reference-wrap": swap(swap(swap(gradients, "model", reference(32, 2, 4)), "benchmark.tail", "wrap"),
                                   "evaluate", Evaluate(every=10, batch=16)),
            "sparsemax-cosine": sparse,
            "amsgradw": swap(swap(base, "diagnostics", TRACE), "optimizer",
                             AMSGradW(lr=1e-3, betas=(0.9, 0.999), weight_decay=1.0)),
            "amsgradw-clipped": swap(swap(base, "diagnostics", TRACE), "optimizer",
                                     Clipped(AMSGradW(lr=1e-3, betas=(0.9, 0.999), weight_decay=1.0), norm=0.01)),
            "sgd": swap(gradients, "optimizer", SGD(lr=0.1, weight_decay=0.01)),
            "adamx": swap(swap(base, "diagnostics", TRACE), "optimizer", AdamX(lr=1e-2)),
            "adamnc": swap(gradients, "optimizer", AdamNC(lr=3e-2)),
            "muon-guarded": swap(swap(base, "diagnostics", TRACE), "optimizer", Guarded(Muon(lr=3e-2))),
            "dash-chebyshev": swap(gradients, "optimizer", Dash(lr=1e-3, solver=Chebyshev())),
            "adafisherw": swap(swap(base, "diagnostics", TRACE), "optimizer", AdaFisher(lr=1e-3, weight_decay=0.01)),
            "magma-adamw": swap(swap(base, "diagnostics", TRACE), "optimizer",
                                Magma(AdamW(lr=1e-3, betas=(0.9, 0.999), weight_decay=0.01))),
            "amsgradmd": swap(gradients, "optimizer", AMSGradMD(lr=3e-3)),
            "amsgradmd-guarded": swap(swap(base, "diagnostics", TRACE), "optimizer",
                                      Guarded(AMSGradMD(lr=0.3, direction_rate=3e-4), sigma=0.25)),
            "text": swap(swap(text(TEXT["runs"]["sparsemax"]), "execution", CudaGraph()), "diagnostics", TRACE),
            "synthetic": swap(swap(synthetic(SYNTHETIC["runs"]["crasp-amsgradw-clipped"]), "execution", CudaGraph()),
                              "diagnostics", TRACE),
            "generated": swap(swap(synthetic(SYNTHETIC["runs"]["parity-running-amsgradw-clipped"]), "execution",
                                   CudaGraph()), "diagnostics", TRACE)}


SIZES = {"reference-wrap": [8], "text": [8], "synthetic": [4, 8], "generated": [4, 8]}


def train(experiment, root, stepper, progress=lambda row: None):
    run = RunDirectory(root)
    if run.description() is None:
        run.begin(describe(experiment), {"engine": {}}, "# a test experiment\n")
    training = Training(experiment, run, progress, stepper)
    training()
    return training.stepper


def summary(value):
    """Tensors as hashes; timings and memory peaks left out."""
    if torch.is_tensor(value):
        return sha(value) if value.ndim else value.item()
    if isinstance(value, dict):
        return {str(key): summary(item) for key, item in value.items()
                if key not in ("training_seconds", "diagnostic_seconds", "wall_seconds",
                               "peak_cuda_allocated_bytes", "peak_cuda_reserved_bytes")}
    if isinstance(value, (list, tuple)):
        return [summary(item) for item in value]
    return value


def records(root):
    run = RunDirectory(root)
    return {**{stream: untimed(run.records(stream)) for stream in STREAMS},
            "result": summary(run.result()), "checkpoint": summary(run.checkpoint("cpu"))}


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
class GraphTests(unittest.TestCase):
    def test_replays_equal_the_operations_issued_eagerly(self):
        for name, experiment in experiments().items():
            with self.subTest(experiment=name), tempfile.TemporaryDirectory() as root:
                replayed, issued = Path(root) / "replayed", Path(root) / "issued"
                stepper = train(experiment, replayed, Counted)
                train(experiment, issued, Issued)
                sampled = len(RunDirectory(replayed).records("diagnostics"))
                self.assertEqual(stepper.issued, sampled)
                self.assertEqual(sorted(stepper.updates), SIZES.get(name, [6, 8]))
                self.assertEqual(json.dumps(records(replayed)), json.dumps(records(issued)))

    def test_an_interrupted_graph_run_resumes_onto_the_same_records(self):
        for name, interrupted in (("sampled", 21), ("text", 25), ("adamx", 21), ("muon-guarded", 21),
                                  ("adafisherw", 21), ("magma-adamw", 21), ("amsgradmd-guarded", 21),
                                  ("synthetic", 21), ("generated", 21)):
            experiment = swap(experiments()[name], "checkpoint", Checkpoint(every=10))

            def interrupt(row):
                if row["step"] == interrupted:
                    raise Interrupted

            with self.subTest(experiment=name), tempfile.TemporaryDirectory() as root:
                straight, resumed = Path(root) / "straight", Path(root) / "resumed"
                train(experiment, straight, GraphStepper)
                with self.assertRaises(Interrupted):
                    train(experiment, resumed, GraphStepper, interrupt)
                train(experiment, resumed, GraphStepper)
                self.assertEqual(json.dumps(records(resumed)), json.dumps(records(straight)))


if __name__ == "__main__":
    unittest.main()
