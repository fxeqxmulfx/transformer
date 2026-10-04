"""A compiled run resumes onto its own records and follows the eager trajectory.

Compiled kernels and the fused AdamW round differently from the eager ones,
so the eager run is a reference within a tolerance; an interrupted compiled
run must resume onto the records of an uninterrupted one bit for bit.
"""

import json
from pathlib import Path
import tempfile
import unittest

from lab.domain.spec import swap
from lab.domain.training import Checkpoint, Compiled, Eager
from lab.infrastructure.engine.compiled import CompiledStepper
from lab.infrastructure.engine.eager import EagerStepper
from lab.infrastructure.store import RunDirectory

from examples import gptmini, modular
from test_engine import TRACE, Interrupted
from test_graphs import records, train
from test_synthetic_training import FIXTURE as SYNTHETIC
from test_synthetic_training import experiment as synthetic


def compiled(experiment):
    return swap(swap(experiment, "execution", Compiled()), "checkpoint", Checkpoint(every=10))


def experiments():
    """Fused AdamW sampled on modular division; clipped AdamW on recall; generated copies; a direction optimizer."""
    return {"modular": swap(compiled(modular(gptmini(32, 2, 4), prime=11, updates=30, batch=8, every=10)),
                            "diagnostics", TRACE),
            "mqar": compiled(synthetic(SYNTHETIC["runs"]["mqar-adamw-clipped"])),
            "copy": compiled(synthetic(SYNTHETIC["runs"]["copy-adamw"])),
            "parity": compiled(synthetic(SYNTHETIC["runs"]["parity-running-amsgradw-clipped"]))}


class CompiledTests(unittest.TestCase):
    def test_an_interrupted_compiled_run_resumes_onto_the_same_records(self):
        for name, experiment in experiments().items():
            def interrupt(row):
                if row["step"] == 20:
                    raise Interrupted

            with self.subTest(experiment=name), tempfile.TemporaryDirectory() as root:
                straight, resumed = Path(root) / "straight", Path(root) / "resumed"
                train(experiment, straight, CompiledStepper)
                with self.assertRaises(Interrupted):
                    train(experiment, resumed, CompiledStepper, interrupt)
                train(experiment, resumed, CompiledStepper)
                self.assertEqual(json.dumps(records(resumed)), json.dumps(records(straight)))

    def test_a_compiled_run_follows_the_eager_one(self):
        for name, experiment in experiments().items():
            with self.subTest(experiment=name), tempfile.TemporaryDirectory() as root:
                train(experiment, Path(root) / "compiled", CompiledStepper)
                train(swap(experiment, "execution", Eager(device="cpu")), Path(root) / "eager", EagerStepper)
                compiled, eager = ([row[name]["loss"] for row in RunDirectory(Path(root) / kind).records("history")
                                    for name in experiment.benchmark.observed if row[name] is not None]
                                   for kind in ("compiled", "eager"))
                self.assertEqual(len(compiled), len(eager))
                for this, that in zip(compiled, eager):
                    self.assertLess(abs(this - that), 1e-4 * abs(that))


if __name__ == "__main__":
    unittest.main()
