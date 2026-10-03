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
from lab.domain.spec import describe, substitute, swap
from lab.domain.training import Checkpoint, Cosine, CudaGraph, Diagnostics, Evaluate
from lab.infrastructure.engine.graphs import GraphStepper
from lab.infrastructure.engine.loop import Training
from lab.infrastructure.store import STREAMS, RunDirectory

from examples import gptmini, modular, reference
from test_engine import TRACE, Interrupted, digest, untimed


class Issued(GraphStepper):
    """Every operation issued eagerly."""

    def prepare(self):
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
            "reference-wrap": swap(swap(swap(gradients, "model", reference(32, 2, 4)), "budget.tail", "wrap"),
                                   "evaluate", Evaluate(every=10, batch=16)),
            "sparsemax-cosine": sparse}


def train(experiment, root, stepper, progress=lambda row: None):
    run = RunDirectory(root)
    if run.description() is None:
        run.begin(describe(experiment), {"engine": {}}, "# a test experiment\n")
    training = Training(experiment, run, progress, stepper)
    training()
    return training.stepper


def records(root):
    run = RunDirectory(root)
    return {**{stream: untimed(run.records(stream)) for stream in STREAMS},
            "checkpoint": digest(run.checkpoint("cpu"))}


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
                self.assertEqual(sorted(stepper.updates), [8] if experiment.budget.tail == "wrap" else [6, 8])
                self.assertEqual(json.dumps(records(replayed)), json.dumps(records(issued)))

    def test_an_interrupted_graph_run_resumes_onto_the_same_records(self):
        experiment = swap(experiments()["sampled"], "checkpoint", Checkpoint(every=10))

        def interrupt(row):
            if row["step"] == 21:
                raise Interrupted

        with tempfile.TemporaryDirectory() as root:
            straight, resumed = Path(root) / "straight", Path(root) / "resumed"
            train(experiment, straight, GraphStepper)
            with self.assertRaises(Interrupted):
                train(experiment, resumed, GraphStepper, interrupt)
            train(experiment, resumed, GraphStepper)
            self.assertEqual(json.dumps(records(resumed)), json.dumps(records(straight)))


if __name__ == "__main__":
    unittest.main()
