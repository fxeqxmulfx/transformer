"""Matched division data, independent loss/decoding checks and live MLP controls."""

from dataclasses import replace
import math
from pathlib import Path
import tempfile
import unittest

import torch
from torch.func import functional_call
from torch.nn import functional as F

from lab.application.report import report_study
from lab.application.study import Study, run_study
from lab.dsl import (MLP, AdamW, Budget, Checkpoint, CrossEntropyAnswers, CudaGraph, Diagnostics,
                     DivisionPairs, Eager, Evaluate, Experiment, LabDivision, ModularRegression,
                     OneHot, Scalar, Schedule, Seeds, SiLU, SquaredAnswers, TorchDefault, VectorReadout)
from lab.infrastructure.benchmarks import build_task
from lab.infrastructure.benchmarks.samplers import gather
from lab.infrastructure.engine import Engine
from lab.infrastructure.nn import build_model
from lab.infrastructure.store import RunDirectories

from test_regression import lookup

CPU = torch.device("cpu")


def network(encoding=Scalar(), width=4):
    return MLP(width=width, depth=2, encoding=encoding, activation=SiLU(), init=TorchDefault(), context=2)


def experiment(vector=False, loss=SquaredAnswers()):
    model = network(OneHot(), width=8)
    return Experiment(model=VectorReadout(model) if vector else model,
                      benchmark=DivisionPairs(LabDivision(prime=5), loss),
                      optimizer=AdamW(lr=0.01, betas=(0.9, 0.98), weight_decay=0),
                      schedule=Schedule(warmup=10, inclusive=True), budget=Budget(updates=6, batch=4),
                      seeds=Seeds(model=11, data=11, batches=11), evaluate=Evaluate(every=3, batch=4),
                      execution=Eager(device="cpu"), diagnostics=Diagnostics(every=3, gradients=True),
                      checkpoint=Checkpoint(every=3))


class DivisionPairTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_all_objectives_keep_source_rows_splits_and_batch_order(self):
        for seed in (11, 17, 23):
            source = build_task(LabDivision(), seed, CPU)
            for vector, loss in ((False, SquaredAnswers()), (True, SquaredAnswers()),
                                 (True, CrossEntropyAnswers())):
                model = VectorReadout(network()) if vector else network()
                task = build_task(DivisionPairs(LabDivision(), loss), seed, CPU, model)
                for split in ("train", "heldout"):
                    torch.testing.assert_close(task.splits[split], source.splits[split], rtol=0, atol=0)
                expected, sampler = source.sampler(512, seed), task.sampler(512, seed)
                for _ in range(21):
                    torch.testing.assert_close(gather(sampler.next()[0]), gather(expected.next()[0]), rtol=0, atol=0)
                rows = torch.cat(tuple(task.splits.values()))
                self.assertTrue(bool(((rows[:, 1] * rows[:, 2] - rows[:, 0]) % 97 == 0).all()))

    def test_hidden_draws_are_shared_and_head_dimensions_are_explicit(self):
        for encoding, counts in ((Scalar(), (39001, 57817)), (OneHot(), (76441, 95257))):
            spec = network(encoding, width=195)
            scalar = build_model(spec, 97, 11)
            vector = build_model(VectorReadout(spec), 97, 11)
            for a, b in zip(scalar.hidden.parameters(), vector.hidden.parameters(), strict=True):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            self.assertEqual(sum(p.numel() for p in scalar.parameters()), counts[0])
            self.assertEqual(sum(p.numel() for p in vector.parameters()), counts[1])
            inputs = torch.tensor([[0, 1], [96, 96]])
            self.assertEqual(scalar(inputs).shape, (2,))
            self.assertEqual(vector(inputs).shape, (2, 97))
            again = build_model(VectorReadout(spec), 97, 11)
            for a, b in zip(vector.parameters(), again.parameters(), strict=True):
                torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_losses_and_numeric_units_match_independent_small_calculations(self):
        rows = torch.tensor([[0, 1, 0], [1, 1, 1], [2, 1, 2]])
        labels = rows[:, 2]
        scalar_spec = network()
        scalar = build_task(DivisionPairs(LabDivision(prime=5), SquaredAnswers()), 11, CPU, scalar_spec)
        predictions = torch.arange(5, dtype=torch.float64) / 2 - 1 + 0.1
        sums = scalar.accumulator()
        scalar.accumulate(lambda inputs: predictions[inputs[:, 0]], rows, sums, False)
        metrics = scalar.metrics(sums.tolist(), rows)
        self.assertAlmostEqual(metrics["loss"], .005, places=13)
        self.assertEqual(metrics["accuracy"], 1)
        self.assertEqual(metrics["numeric_mse"], 0)
        self.assertAlmostEqual(metrics["continuous_numeric_mse"], .04, places=13)
        for objective in (SquaredAnswers(), CrossEntropyAnswers()):
            task = build_task(DivisionPairs(LabDivision(prime=5), objective), 11, CPU, VectorReadout(scalar_spec))
            output = F.one_hot(labels, 5).double()
            expected = 0 if isinstance(objective, SquaredAnswers) else math.log(math.e + 4) - 1
            self.assertAlmostEqual(float(task.loss(output, labels)), expected, places=13)
            if isinstance(objective, SquaredAnswers):
                self.assertEqual(float(task.loss(torch.zeros_like(output), labels)), .5)
            sums = task.accumulator()
            task.accumulate(lambda inputs: F.one_hot(inputs[:, 0], 5).double(), rows, sums, False)
            metrics = task.metrics(sums.tolist(), rows)
            self.assertEqual(metrics["accuracy"], 1)
            self.assertEqual(metrics["numeric_mse"], 0)

    def test_every_parameter_has_the_correct_loss_gradient_in_float64(self):
        inputs, labels = torch.tensor([[0, 1], [2, 3], [4, 2]]), torch.tensor([0, 4, 2])
        for encoding in (Scalar(), OneHot()):
            for vector, objective in ((False, SquaredAnswers()), (True, SquaredAnswers()),
                                      (True, CrossEntropyAnswers())):
                with self.subTest(encoding=encoding, vector=vector, objective=objective):
                    spec = VectorReadout(network(encoding)) if vector else network(encoding)
                    model = build_model(spec, 5, 11).double()
                    task = build_task(DivisionPairs(LabDivision(prime=5), objective), 11, CPU, spec)
                    names, parameters = zip(*model.named_parameters(), strict=True)

                    def loss(*values):
                        state = dict(zip(names, values, strict=True))
                        return task.loss(functional_call(model, state, (inputs,)), labels)

                    self.assertTrue(torch.autograd.gradcheck(loss, parameters))
                    loss(*parameters).backward()
                    for p in parameters:
                        self.assertIsNotNone(p.grad)
                        self.assertGreater(float(p.grad.norm()), 0)

    def test_finite_scalar_silu_weights_answer_every_division_row(self):
        full = build_task(ModularRegression(prime=97, operation="/"), 0, CPU)
        spec = network(width=195)
        model = build_model(spec, 97, 11).double()
        lookup(model, full, gain=30)
        task = build_task(DivisionPairs(LabDivision(), SquaredAnswers()), 11, CPU, spec)
        for rows in task.splits.values():
            sums = task.accumulator()
            with torch.no_grad():
                task.accumulate(model, rows, sums, False)
            metrics = task.metrics(sums.tolist(), rows)
            self.assertEqual(metrics["accuracy"], 1)
            self.assertLess(metrics["continuous_numeric_mse"], 1e-8)

    def test_real_engine_trains_and_checkpoints_each_objective(self):
        for vector, loss in ((False, SquaredAnswers()), (True, SquaredAnswers()),
                             (True, CrossEntropyAnswers())):
            spec = experiment(vector, loss)
            with tempfile.TemporaryDirectory() as directory:
                runs = RunDirectories(Path(directory))
                study = Study("paired", "Matched readout integration", {"run": spec})
                result = run_study(study, [], runs, Engine(), lambda label, row: None)["run"]
                history = runs.open("run").records("history")
                report = report_study(study, [], runs)
                self.assertIn("run", report["runs"])
                self.assertEqual([p["step"] for p in history], [0, 3, 6])
                self.assertEqual(runs.open("run").checkpoint(CPU)["step"], 6)
                self.assertEqual(result["stop"]["reason"], "budget")
                self.assertEqual(history[-1]["heldout"]["examples"], 10)
                self.assertIn("numeric_mse", history[-1]["heldout"])
                self.assertIn("first_activation_rms", history[-1])
                initial = build_model(spec.model, 5, spec.seeds.model)
                self.assertTrue(any(not torch.equal(p, runs.open("run").checkpoint(CPU)["model"][name])
                                    for name, p in initial.named_parameters()))

    def test_invalid_output_objective_and_execution_compositions_fail(self):
        base = experiment()
        with self.assertRaises(ValueError):
            replace(base, benchmark=DivisionPairs(LabDivision(prime=5), CrossEntropyAnswers()))
        with self.assertRaises(ValueError):
            replace(base, execution=CudaGraph())
        with self.assertRaises(ValueError):
            replace(base, model=replace(base.model, context=3))
        with self.assertRaises(ValueError):
            build_task(base.benchmark, 11, CPU)
