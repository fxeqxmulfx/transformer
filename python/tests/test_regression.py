"""Check the modular regression table, the perceptron, weights answering every row, the landscape
measurements, the step against the curvature, and a real run with its report.

The Hessian extremes are checked against the eigenvalues of the dense
Hessian of a small perceptron, and the spectrum shares against a table whose
transform is flat.
"""

from dataclasses import replace
from pathlib import Path
import math
import tempfile
import unittest

import torch
from torch import nn
from torch.func import functional_call

from lab.application.report import curvature, report_study
from lab.application.study import Study, run_study
from lab.domain.analysis import decade_slope, step_curvature
from lab.dsl import (GELU, MLP, SGD, AdamW, Budget, Checkpoint, Cosine, CudaGraph, Eager, Evaluate, Experiment,
                     ModularDivision, ModularRegression, OneHot, ReLU, Scalar, ScaledResidual, Schedule, Seeds,
                     TorchDefault, swap)
from lab.infrastructure.benchmarks import build_task
from lab.infrastructure.benchmarks.regression import high_frequency_shares, lanczos
from lab.infrastructure.engine import Engine
from lab.infrastructure.nn import build_model
from lab.infrastructure.store import RunDirectories

from examples import reference

CPU = torch.device("cpu")


def perceptron(width=8, depth=2, encoding=Scalar(), activation=GELU()):
    return MLP(width=width, depth=depth, encoding=encoding, activation=activation, init=TorchDefault(), context=2)


def regression():
    return Experiment(model=perceptron(), benchmark=ModularRegression(prime=5, operation="/"),
                      optimizer=SGD(lr=0.1), schedule=Schedule(), budget=Budget(updates=20, batch=20),
                      seeds=Seeds(), evaluate=Evaluate(every=10, batch=7), execution=Eager(device="cpu"),
                      checkpoint=Checkpoint(every=10))


def lookup(model, task, gain):
    """Weights under which a scalar perceptron of depth 2 and width at least 2p + 1 answers every row.

    With t = (p - 1)(s + 1) / 2 the operand s decoded, the first layer holds
    ReLU(t - k) for x at k = -1, ..., p - 1 and for y at k = 0, ..., p - 1;
    their combinations ReLU(t - a + 1) - 2 ReLU(t - a) + ReLU(t - a - 1) are
    the indicators h_a of the integers a. For each x value a the second layer
    holds ReLU(2 h_a(x) - 2 +- g_a(y)), g_a(y) = sum_b target(a, b) h_b(y),
    and the output adds their difference, h_a(x) g_a(y), over a. Under GELU
    the first layer's weights and both biases are multiplied by `gain` and
    the output divided by it. The first layer's preactivations are then
    multiples of `gain`, and the second's at most -gain where h_a(x) = 0,
    where GELU is ReLU up to gain Phi(-gain); where h_a(x) = 1 the pair's
    difference is exact, GELU(z) - GELU(-z) = z.
    """
    prime, (first, second), width = task.vocab, model.hidden, model.spec.width
    rows = task.splits["train"].cpu()
    table = torch.zeros(prime, prime, dtype=torch.float64)
    table[rows[:, 0], rows[:, 1]] = rows[:, 2].double() * (2 / (prime - 1)) - 1
    thresholds = [(0, k) for k in range(-1, prime)] + [(1, k) for k in range(prime)]
    weight = torch.zeros(width, 2, dtype=torch.float64)
    bias = torch.zeros(width, dtype=torch.float64)
    for unit, (operand, k) in enumerate(thresholds):
        weight[unit, operand], bias[unit] = (prime - 1) / 2, (prime - 1) / 2 - k
    hats = torch.zeros(2, prime, width, dtype=torch.float64)
    for unit, (operand, k) in enumerate(thresholds):
        for value, coefficient in ((k + 1, 1.0), (k, -2.0), (k - 1, 1.0)):
            if 0 <= value < prime:
                hats[operand, value, unit] = coefficient
    answers = table @ hats[1]
    middle = torch.zeros(width, width, dtype=torch.float64)
    readout = torch.zeros(1, width, dtype=torch.float64)
    for a in range(prime):
        for unit, sign in ((2 * a, 1.0), (2 * a + 1, -1.0)):
            middle[unit], readout[0, unit] = 2 * hats[0, a] + sign * answers[a], sign
    with torch.no_grad():
        first.weight.copy_(gain * weight)
        first.bias.copy_(gain * bias)
        second.weight.copy_(middle)
        second.bias.copy_(torch.where(readout[0] != 0, -2.0 * gain, 0.0))
        model.output.weight.copy_(readout / gain)
        model.output.bias.zero_()


class RegressionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_rows_are_the_whole_table_and_obey_the_oracle(self):
        prime = 7
        for operation in ("+", "/"):
            rows = build_task(ModularRegression(prime=prime, operation=operation), 0, CPU).splits["train"].tolist()
            self.assertEqual([(x, y) for x, y, _ in rows], [(x, y) for x in range(prime) for y in range(1, prime)])
            for x, y, c in rows:
                self.assertEqual((x + y - c) % prime if operation == "+" else (c * y - x) % prime, 0)
            self.assertEqual(sorted(c for _, _, c in rows), sorted(list(range(prime)) * (prime - 1)))

    def test_encodings_read_the_operands(self):
        scalar = build_model(perceptron(), 7, 0)
        torch.testing.assert_close(scalar.encode(torch.tensor([[0, 6], [3, 3]])),
                                   torch.tensor([[-1.0, 1.0], [0.0, 0.0]]))
        onehot = build_model(perceptron(encoding=OneHot()), 7, 0).encode(torch.tensor([[2, 3]]))
        self.assertEqual(onehot.shape, (1, 14))
        self.assertEqual(onehot.nonzero()[:, 1].tolist(), [2, 10])

    def test_a_seed_draws_the_parameters_of_the_sequential_perceptron(self):
        built = build_model(perceptron(), 7, 3)
        torch.manual_seed(3)
        sequential = nn.Sequential(nn.Linear(2, 8), nn.GELU(), nn.Linear(8, 8), nn.GELU(), nn.Linear(8, 1))
        for ours, theirs in zip(built.parameters(), sequential.parameters(), strict=True):
            torch.testing.assert_close(ours, theirs, rtol=0, atol=0)
        inputs = torch.tensor([[1, 2], [6, 5]])
        torch.testing.assert_close(built(inputs), sequential(built.encode(inputs)).squeeze(-1), rtol=0, atol=0)

    def test_width_2p_plus_1_answers_every_row_of_either_table(self):
        for operation in ("/", "+"):
            task = build_task(ModularRegression(prime=97, operation=operation), 0, CPU)
            for activation, gain in ((ReLU(), 1.0), (GELU(), 10.0)):
                model = build_model(perceptron(width=195, activation=activation), task.vocab, 0)
                lookup(model, task, gain)
                sums = task.accumulator()
                with torch.no_grad():
                    task.accumulate(model, task.splits["train"], sums, False)
                metrics = task.metrics(sums.tolist(), task.splits["train"])
                self.assertEqual(metrics["accuracy"], 1.0)
                self.assertLess(metrics["loss"], 1e-8)

    def test_chunked_gradient_and_hessian_products_match_the_whole_table(self):
        task = build_task(ModularRegression(prime=5, operation="/"), 0, CPU)
        model = build_model(perceptron(), task.vocab, 1)
        parameters = list(model.parameters())
        output, targets = task.forward(model, task.splits["train"])
        whole = torch.cat([g.reshape(-1) for g in torch.autograd.grad(task.loss(output, targets), parameters)])
        torch.testing.assert_close(task.gradient(model, parameters, 7), whole)
        vector = torch.randn(len(whole), generator=torch.Generator().manual_seed(2))
        torch.testing.assert_close(task.gradient(model, parameters, 7, vector),
                                   task.gradient(model, parameters, 100, vector))

    def test_lanczos_extremes_are_the_dense_hessian_extremes(self):
        task = build_task(ModularRegression(prime=5, operation="+"), 0, CPU)
        for activation in (GELU(), ReLU()):
            model = build_model(perceptron(width=4, depth=1, activation=activation), task.vocab, 4)
            names = [name for name, _ in model.named_parameters()]
            shapes = [parameter.shape for parameter in model.parameters()]
            point = torch.cat([parameter.detach().reshape(-1) for parameter in model.parameters()])

            def loss(vector):
                pieces = vector.split([math.prod(shape) for shape in shapes])
                state = {name: piece.reshape(shape) for name, piece, shape in zip(names, pieces, shapes)}
                output = functional_call(model, state, (task.splits["train"][:, :2],))
                return task.loss(output, task.forward(model, task.splits["train"])[1])

            dense = torch.linalg.eigvalsh(torch.autograd.functional.hessian(loss, point))
            parameters = list(model.parameters())
            ritz = lanczos(lambda vector: task.gradient(model, parameters, 7, vector), len(point), len(point),
                           torch.Generator().manual_seed(0))
            scale = dense.abs().max().item()
            self.assertAlmostEqual(ritz[0], dense[0].item(), delta=1e-4 * scale)
            self.assertAlmostEqual(ritz[-1], dense[-1].item(), delta=1e-4 * scale)

    def test_a_point_table_spreads_its_energy_evenly(self):
        prime = 7
        residual = torch.zeros(prime * (prime - 1), dtype=torch.float64)
        residual[9] = 1.0
        # Distances 0, 1, 2, 3, 3, 2, 1: 3 x 3 frequencies have level at most 1, 5 x 5 at most 2.
        shares = high_frequency_shares(residual, prime, (1, 2, 3))
        self.assertAlmostEqual(shares["1"], 40 / 49)
        self.assertAlmostEqual(shares["2"], 24 / 49)
        self.assertAlmostEqual(shares["3"], 0.0)
        self.assertEqual(high_frequency_shares(torch.zeros_like(residual), prime, (1,)), {"1": None})

    def test_decade_slope_reads_the_last_decade(self):
        self.assertAlmostEqual(decade_slope([(step, 3 * step ** -0.5) for step in range(50, 1001, 50)]), -0.5)
        self.assertAlmostEqual(decade_slope([(step, 2.0) for step in (100, 500, 1000)]), 0.0)
        self.assertAlmostEqual(decade_slope([(1, 9.0), (10, 9.0), (100, 1.0), (1000, 0.1)]), -1.0)
        self.assertIsNone(decade_slope([(0, 1.0), (1000, 1.0)]))

    def test_step_curvature_finds_the_first_step_reaching_one(self):
        history = [{"step": step, "landscape": {"hessian_min": low, "hessian_max": high}}
                   for step, low, high in ((0, -0.5, 4.0), (1000, -8.0, 6.0), (2000, 1.0, 10.0))]
        found = step_curvature(history, 0.125)
        self.assertEqual(found["values"], [[0, 0.5], [1000, 1.0], [2000, 1.25]])
        self.assertEqual((found["max"], found["first_step_reaching_one"]), (1.25, 1000))
        self.assertIsNone(step_curvature(history[:1], 0.125)["first_step_reaching_one"])
        self.assertIsNone(step_curvature([], 0.125)["max"])

    def test_only_a_perceptron_fits_the_regression_eagerly(self):
        base = regression()
        with self.assertRaises(ValueError):
            swap(base, "model", reference())
        with self.assertRaises(ValueError):
            replace(base, model=swap(perceptron(), "context", 6), benchmark=ModularDivision(prime=5, train_fraction=0.5))
        with self.assertRaises(ValueError):
            swap(base, "execution", CudaGraph())
        with self.assertRaises(ValueError):
            swap(base, "model.context", 3)
        for wrong in ({"operation": "*"}, {"prime": 9}):
            with self.assertRaises(ValueError):
                ModularRegression(**{"prime": 5, "operation": "/", **wrong})
        with self.assertRaises(ValueError):
            swap(base, "model.init", ScaledResidual())

    def test_training_descends_and_records_the_landscape(self):
        with tempfile.TemporaryDirectory() as root:
            study = Study("regression", "Modular regression check", {"run": regression()})
            result = run_study(study, [], RunDirectories(Path(root)), Engine(), lambda label, row: None)["run"]
            history = RunDirectories(Path(root)).open("run").records("history")
            report = report_study(study, [], RunDirectories(Path(root)))["runs"]["run"]
        self.assertEqual([row["step"] for row in history], [0, 10, 20])
        self.assertLess(history[-1]["train"]["loss"], history[0]["train"]["loss"])
        landscape = result["last"]["landscape"]
        self.assertEqual(landscape["lanczos_iterations"], 64)
        self.assertLessEqual(landscape["hessian_min"], landscape["hessian_max"])
        self.assertEqual(set(history[0]["landscape"]["residual_high"]), {"5", "10", "20"})
        self.assertIn("loss", result["final_decade_slopes"])
        values = [[point["step"], 0.1 * max(abs(point["landscape"]["hessian_min"]), point["landscape"]["hessian_max"])]
                  for point in history]
        self.assertEqual(report["step_curvature"]["values"], values)
        self.assertEqual(report["step_curvature"]["max"], max(value for _, value in values))
        self.assertEqual(report["step_curvature"]["first_step_reaching_one"],
                         next((step for step, value in values if value >= 1), None))
        controls = (swap(regression(), "optimizer", AdamW(lr=1e-3, betas=(0.9, 0.999), weight_decay=0.0)),
                    swap(regression(), "optimizer.weight_decay", 0.1), swap(regression(), "budget.batch", 7),
                    swap(regression(), "schedule.warmup", 5),
                    swap(regression(), "schedule.anneal", Cosine(start=0, end=20, final=0.1)))
        for control in controls:
            self.assertIn("not_applicable", curvature(control, history))


if __name__ == "__main__":
    unittest.main()
