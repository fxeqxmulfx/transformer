"""The eager engine reproduces the historical text trainer, observation for observation.

Golden records: the `runs` and `cuda_runs` of `fixtures/legacy_text.json`,
written by `full_compile_benchmark.training.train_one` with its step
uncompiled, under SGD, at the commit the fixture names: every validation
loss, why and where the run stopped, the best observation, the test loss of
its model, and that model's parameters.

On CUDA the historical step differentiated through `torch.func.grad_and_value`,
whose kernels round otherwise than autograd's `backward` (on the CPU the two
agree bit for bit). The first observation, which involves no gradient, is
reproduced exactly; from there the trajectories part in the last bits, which
sparsemax amplifies to 2.4e-4 relative over 40 updates (the historical
trainer's own CPU and CUDA runs part by 8.2e-5), and every decision is the same.
"""

import json
import math
from pathlib import Path
import tempfile
import unittest

import torch

from lab.domain.model import Softmax, Sparsemax
from lab.domain.spec import substitute, swap
from lab.dsl import (SGD, Budget, Checkpoint, EarlyStopping, Eager, Evaluate, Experiment, Schedule, Seeds,
                     TinyShakespeare)
from lab.infrastructure.benchmarks.samplers import WindowSampler, gather
from lab.infrastructure.nn.legacy import rename
from lab.infrastructure.store import RunDirectory

from examples import gptmini
from test_engine import Interrupted, sha, train

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "legacy_text.json").read_text())
REASONS = {"max_steps": "budget", "patience": "patience", "validation_divergence": "divergence",
           "nonfinite_validation": "nonfinite_selection"}


def historical(golden, device="cpu"):
    """A fixture run in the DSL: GPTMini of width 32 on windows of 16 characters, under SGD."""
    config, stop = golden["config"], golden["config"]["stop"]
    model = swap(gptmini(32, 2, 4), "context", config["model"]["max_seq_len"])
    if config["attention"] == "sparsemax":
        model = substitute(model, Softmax, Sparsemax())
    return Experiment(
        model=model, benchmark=TinyShakespeare(window=config["model"]["max_seq_len"]),
        optimizer=SGD(lr=config["rate"]), schedule=Schedule(),
        budget=Budget(updates=stop["max_steps"], batch=config["batch"]),
        seeds=Seeds(model=config["seed"], data=0), evaluate=Evaluate(every=stop["every"], batch=config["batch"]),
        stopping=EarlyStopping(patience=stop["patience"], min_delta=stop["min_delta"], after=stop["min_steps"],
                               divergence=stop["divergence_delta"], divergence_patience=stop["divergence_patience"]),
        execution=Eager(device=device))


class WindowSamplerTests(unittest.TestCase):
    def test_starts_drawn_in_blocks_equal_the_starts_drawn_at_once(self):
        expected = torch.randint(1000, (10, 4), generator=torch.Generator().manual_seed(10_007))
        sampler, drawn = WindowSampler(1000, 4, 10_007, block=3), []
        for update in range(10):
            if update == 5:
                state, sampler = sampler.state(), WindowSampler(1000, 4, 0, block=3)
                sampler.restore(state)
            drawn.append(gather(sampler.next()[0]))
        self.assertTrue(torch.equal(torch.stack(drawn), expected))


def reproduces(test, golden, root):
    """The run under `root` makes the golden observations, decisions and best model, bit for bit."""
    run = RunDirectory(Path(root) / "run")
    test.assertEqual([(row["step"], row["validation"]["loss"]) for row in run.records("history")],
                     [(check["step"], check["validation_loss"]) for check in golden["curves"]])
    result = run.result()
    stopped = golden["stop_reason"] != "max_steps"
    test.assertEqual(result["stop"], {"step": golden["actual_steps"] if stopped else result["updates"],
                                      "reason": REASONS[golden["stop_reason"]]})
    best = result["best"]
    test.assertEqual((best["step"], best["validation"]["loss"], best["test"]["loss"]),
                     (golden["best_step"], golden["validation_loss"], golden["test_loss"]))
    checkpoint = run.checkpoint("cpu")
    test.assertEqual(checkpoint["step"], golden["actual_steps"])
    test.assertEqual({name: sha(value) for name, value in checkpoint["best"]["model"].items()},
                     {rename(name): value for name, value in golden["best"].items()})


class HistoricalTextTests(unittest.TestCase):
    def assert_reproduces(self, golden, root):
        reproduces(self, golden, root)

    def test_every_historical_cpu_run(self):
        for name, golden in FIXTURE["runs"].items():
            with self.subTest(run=name), tempfile.TemporaryDirectory() as root:
                train(historical(golden), root)
                self.assert_reproduces(golden, root)

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_every_historical_cuda_run_on_the_fixture_gpu_up_to_rounding(self):
        if FIXTURE["source"].get("gpu") != torch.cuda.get_device_name():
            self.skipTest("CUDA golden records were written on another GPU")
        for name, golden in FIXTURE["cuda_runs"].items():
            with self.subTest(run=name), tempfile.TemporaryDirectory() as root:
                result = train(historical(golden, device="cuda"), root)
                history = RunDirectory(Path(root) / "run").records("history")
                self.assertEqual([row["step"] for row in history], [check["step"] for check in golden["curves"]])
                self.assertEqual(history[0]["validation"]["loss"], golden["curves"][0]["validation_loss"])
                for row, check in zip(history, golden["curves"], strict=True):
                    self.assertLess(abs(row["validation"]["loss"] / check["validation_loss"] - 1), 1e-3)
                self.assertEqual((result["stop"]["reason"], result["best"]["step"]),
                                 (REASONS[golden["stop_reason"]], golden["best_step"]))
                self.assertLess(abs(result["best"]["test"]["loss"] / golden["test_loss"] - 1), 1e-3)

    def test_an_interrupted_run_resumes_onto_the_same_records(self):
        golden = FIXTURE["runs"]["softmax"]
        experiment = swap(historical(golden), "checkpoint", Checkpoint(every=10))

        def interrupt(label, row):
            if row["step"] == 25:
                raise Interrupted

        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(Interrupted):
                train(experiment, root, interrupt)
            self.assertEqual(RunDirectory(Path(root) / "run").checkpoint("cpu")["step"], 20)
            train(experiment, root)
            self.assert_reproduces(golden, root)

    def test_a_stopped_run_stays_stopped_under_a_larger_budget(self):
        golden = FIXTURE["runs"]["patience"]
        experiment = historical(golden)
        with tempfile.TemporaryDirectory() as root:
            first = train(experiment, root)
            second = train(swap(experiment, "budget.updates", 60), root)
            self.assertEqual(second["updates"], 60)
            self.assertEqual({key: second[key] for key in ("stop", "best", "final")},
                             {key: first[key] for key in ("stop", "best", "final")})
            self.assert_reproduces(golden, root)

    def test_a_nonfinite_gradient_ends_a_stopped_run_and_fails_any_other(self):
        experiment = swap(historical(FIXTURE["runs"]["softmax"]), "optimizer.lr", 1e30)
        with tempfile.TemporaryDirectory() as root:
            result = train(experiment, root)
            self.assertEqual(result["stop"]["reason"], "nonfinite_gradient")
            self.assertEqual(result["best"]["step"], 0)
            self.assertTrue(math.isfinite(result["best"]["test"]["loss"]))
        with tempfile.TemporaryDirectory() as root, self.assertRaises(FloatingPointError):
            train(swap(experiment, "stopping", None), root)


if __name__ == "__main__":
    unittest.main()
