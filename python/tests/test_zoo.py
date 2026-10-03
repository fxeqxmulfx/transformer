"""The direction optimizers reproduce the historical optimizer zoo on the CPU.

Golden records: the `runs` of `fixtures/legacy_zoo.json`, written by the
`train_one` of the all24 benchmark and of the AMSGrad extensions with the step
uncompiled, for every recipe on GPTMini of width 32, at the top rate of its
grid, or the middle one where the top rate diverges within 40 updates.

A rule reproduces its run bit for bit when it computes as the historical step
did, the coefficients that change with the update included: both compute them
from a float64 update count. A first moment with a constant coefficient is
the exception. It is averaged as the modular trainer averaged it,
`m.mul_(b).add_(g, alpha=1 - b)`, where the historical step wrote
`m.mul_(b).add_(g * (1 - b))`: under the historical form Adam and AMSGrad
reproduce their runs bit for bit, and under their own they part in the last
bits, which Adam's normalization amplifies to 3.1e-5 relative over 40 updates
(AMSGrad 1.3e-5). The zoo's bias-corrected AdamW is PyTorch's, within 9e-8.
They are held to a relative 1e-4 of every observation, with the same decisions.
"""

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from lab.domain.spec import swap
from lab.domain.training import Checkpoint
from lab.dsl import SGD, AdaGrad, Adam, AdamNC, AdamW, AdamX, AMSGradW, Geometric, Inverse, InverseSqrt, RMSProp
from lab.infrastructure.optim import coordinate
from lab.infrastructure.store import RunDirectory

from test_engine import Interrupted, train
from test_text import REASONS, historical, reproduces

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "legacy_zoo.json").read_text())
BETAS = (0.9, 0.999)
EXACT = {
    "sgd": lambda rate: SGD(lr=rate),
    "adagrad": lambda rate: AdaGrad(lr=rate),
    "amsgrad_inverse": lambda rate: AMSGradW(lr=rate, betas=BETAS, weight_decay=0.0,
                                             beta1_decay=Inverse(), lr_decay=InverseSqrt()),
    "amsgrad_geometric": lambda rate: AMSGradW(lr=rate, betas=BETAS, weight_decay=0.0,
                                               beta1_decay=Geometric(0.99), lr_decay=InverseSqrt()),
    "adamx": lambda rate: AdamX(lr=rate),
    "adamnc": lambda rate: AdamNC(lr=rate),
    "rmsprop": lambda rate: RMSProp(lr=rate),
}
CLOSE = {
    "adam": lambda rate: Adam(lr=rate),
    "amsgrad": lambda rate: AMSGradW(lr=rate, betas=BETAS, weight_decay=0.0),
    "adamw": lambda rate: AdamW(lr=rate, betas=BETAS, weight_decay=0.01),
}


def recipe(name, recipes):
    golden = FIXTURE["runs"][name]
    return golden, swap(historical(golden), "optimizer", recipes[name](golden["config"]["rate"]))


def historical_average(m, gradient, beta):
    """The first moment as `full_compile_benchmark.optimizer.coordinate` averaged it."""
    m.mul_(beta).add_(gradient * (1 - beta))


class ZooTests(unittest.TestCase):
    def test_rules_reproduce_their_runs(self):
        for name in EXACT:
            golden, experiment = recipe(name, EXACT)
            with self.subTest(recipe=name), tempfile.TemporaryDirectory() as root:
                train(experiment, root)
                reproduces(self, golden, root)

    def test_an_interrupted_run_resumes_onto_the_same_records(self):
        golden, experiment = recipe("adamnc", EXACT)

        def interrupt(label, row):
            if row["step"] == 25:
                raise Interrupted

        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(Interrupted):
                train(swap(experiment, "checkpoint", Checkpoint(every=10)), root, interrupt)
            run = RunDirectory(Path(root) / "historical" / "run")
            self.assertEqual(run.checkpoint("cpu")["optimizer"]["state"][0]["step"].item(), 20)
            train(swap(experiment, "checkpoint", Checkpoint(every=10)), root)
            reproduces(self, golden, root)

    def test_constant_momenta_reproduce_their_runs_under_the_historical_arithmetic(self):
        for name in ("adam", "amsgrad"):
            golden, experiment = recipe(name, CLOSE)
            with (self.subTest(recipe=name), tempfile.TemporaryDirectory() as root,
                  mock.patch.object(coordinate, "average", historical_average)):
                train(experiment, root)
                reproduces(self, golden, root)

    def test_rules_of_other_arithmetic_follow_their_runs(self):
        for name in CLOSE:
            golden, experiment = recipe(name, CLOSE)
            with self.subTest(recipe=name), tempfile.TemporaryDirectory() as root:
                result = train(experiment, root)
                history = RunDirectory(Path(root) / "historical" / "run").records("history")
                self.assertEqual([row["step"] for row in history], [check["step"] for check in golden["curves"]])
                for row, check in zip(history, golden["curves"], strict=True):
                    self.assertLess(abs(row["validation"]["loss"] / check["validation_loss"] - 1), 1e-4)
                self.assertEqual((result["stop"]["reason"], result["best"]["step"]),
                                 (REASONS[golden["stop_reason"]], golden["best_step"]))
                self.assertLess(abs(result["best"]["test"]["loss"] / golden["test_loss"] - 1), 1e-4)


if __name__ == "__main__":
    unittest.main()
