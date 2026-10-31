"""Arithmetic certificates, independent current-model gradients and compiled replay."""

import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

import torch
from torch.func import functional_call
from torch.nn import functional as F

from lab.dsl import (AdamW, Budget, Checkpoint, Eager, Evaluate, Experiment, MaxAutotune,
                     PeriRotate90, PowerDivision, Schedule, Seeds, swap)
from lab.infrastructure.benchmarks import build_task
from lab.infrastructure.engine.compiled import CompiledStepper
from lab.infrastructure.nn import build_model
from lab.infrastructure.store import RunDirectory, STREAMS
from test_engine import Interrupted
from test_graphs import train


def recipe(device="cpu"):
    return Experiment(model=PeriRotate90(8, 2), benchmark=PowerDivision(7, 2),
                      optimizer=AdamW(lr=.003, betas=(.9, .98), weight_decay=.1),
                      budget=Budget(12, 8), seeds=Seeds(11, 11, 11), schedule=Schedule(warmup=2),
                      evaluate=Evaluate(4, 8), checkpoint=Checkpoint(4),
                      execution=MaxAutotune(device=device))


class PowerDivisionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_initial_weights_and_logits_match_the_original_source_fixture(self):
        fixture = json.loads((Path(__file__).parent / "fixtures/peri_rotary_source.json").read_text())
        inputs = torch.tensor(fixture["inputs"])
        for case in fixture["cases"]:
            model = build_model(PeriRotate90(case["width"], case["depth"]), case["prime"], case["seed"])
            flat = torch.cat([p.detach().flatten() for p in model.parameters()])
            digest = hashlib.sha256(flat.numpy().astype("<f4").tobytes()).hexdigest()
            self.assertEqual(digest, case["initial_sha256"])
            torch.testing.assert_close(model(inputs), torch.tensor(case["logits"]), rtol=0, atol=0)

    def test_every_label_has_an_independent_multiplication_certificate(self):
        for prime in (2, 7, 17, 97, 193):
            operands = None
            for exponent in (0, 1, 2, 3, prime + 2):
                task = build_task(PowerDivision(prime, exponent), 11, torch.device("cpu"))
                rows = torch.cat(tuple(task.splits.values()))
                current = rows[:, :2]
                if operands is not None:
                    torch.testing.assert_close(current, operands, rtol=0, atol=0)
                operands = current
                self.assertEqual(len(set(map(tuple, current.tolist()))), prime * (prime - 1))
                self.assertTrue(all(pow(b, exponent, prime) * c % prime == pow(a, exponent, prime)
                                    for a, b, c in rows.tolist()))
                if exponent == 0:
                    self.assertTrue(bool((rows[:, 2] == 1).all()))
                if exponent > 0:
                    self.assertTrue(bool((rows[rows[:, 0] == 0, 2] == 0).all()))

    def test_full_table_is_explicitly_a_fitting_control(self):
        spec = PowerDivision(7, 3, 1.)
        task = build_task(spec, 11, torch.device("cpu"))
        self.assertEqual(spec.observed, ("train",))
        self.assertEqual(len(task.splits["train"]), 42)
        self.assertEqual(len(task.splits["heldout"]), 0)
        self.assertIsNone(task.summary()["heldout_majority_accuracy"])
        self.assertEqual(task.summary()["label_classes"], 3)

    def test_forward_and_gradients_match_independent_matrix_rotation_and_norms(self):
        inputs = torch.tensor([[0, 1], [2, 3], [4, 2]])
        rotation = torch.tensor([[0., 1., 0., 0.], [-1., 0., 0., 0.],
                                 [0., 0., 0., 1.], [0., 0., -1., 0.]], dtype=torch.float64)
        for depth in (1, 3):
            model = build_model(PeriRotate90(4, depth), 5, 11).double()
            weights = {name: p.detach().clone().requires_grad_() for name, p in model.named_parameters()}
            embedding = weights["E"]

            def norm(x):
                return x / (x.square().mean(-1, keepdim=True) + torch.finfo(x.dtype).eps).sqrt()

            x = norm(embedding[inputs[:, 0]] + embedding[inputs[:, 1]] @ rotation)
            for block in range(depth):
                both = norm(x) @ weights[f"gate_value.{block}"]
                gate, value = both[:, :model.spec.hidden], both[:, model.spec.hidden:]
                x = x + norm((gate * torch.sigmoid(gate) * value) @ weights[f"output.{block}"])
            expected = norm(x) @ embedding.T
            actual = model(inputs)
            torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-11)
            targets = torch.tensor([0, 1, 4])
            F.cross_entropy(actual, targets).backward()
            gradients = torch.autograd.grad(F.cross_entropy(expected, targets), tuple(weights.values()))
            for parameter, gradient in zip(model.parameters(), gradients, strict=True):
                torch.testing.assert_close(parameter.grad, gradient, rtol=1e-9, atol=1e-9)
            self.assertEqual(sum(p.numel() for p in model.parameters()),
                             5 * 4 + depth * 3 * 4 * model.spec.hidden)

    def test_every_current_model_parameter_passes_finite_differences(self):
        model = build_model(PeriRotate90(4, 2), 5, 11).double()
        names, parameters = zip(*model.named_parameters(), strict=True)
        inputs = torch.tensor([[0, 1], [2, 3], [4, 2]])

        def evaluate(*values):
            return functional_call(model, dict(zip(names, values, strict=True)), (inputs,))

        self.assertTrue(torch.autograd.gradcheck(evaluate, parameters, fast_mode=True))

    def test_invalid_specs(self):
        for width, depth in ((3, 1), (0, 1), (8, 0)):
            with self.assertRaises(ValueError):
                PeriRotate90(width, depth)
        for prime, exponent, fraction in ((4, 1, .5), (7, -1, .5), (7, 1, 0)):
            with self.assertRaises(ValueError):
                PowerDivision(prime, exponent, fraction)


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
class PowerDivisionCompileTests(unittest.TestCase):
    def test_max_autotune_forward_and_backward_match_eager_on_gpu(self):
        torch.set_num_threads(1)
        for depth in (1, 2):
            torch._dynamo.reset()
            model = build_model(PeriRotate90(16, depth), 17, 11).cuda()
            eager = copy.deepcopy(model)
            inputs = torch.tensor([[0, 1], [2, 3], [4, 2], [16, 7]], device="cuda")
            targets = torch.tensor([0, 3, 4, 7], device="cuda")

            def loss(batch, labels):
                return F.cross_entropy(model(batch), labels)

            compiled = torch.compile(loss, mode="max-autotune", fullgraph=True, dynamic=False)
            actual = compiled(inputs, targets)
            expected = F.cross_entropy(eager(inputs), targets)
            actual.backward()
            expected.backward()
            torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
            for first, second in zip(model.parameters(), eager.parameters(), strict=True):
                torch.testing.assert_close(first.grad, second.grad, rtol=3e-4, atol=3e-4)

    def test_compiled_training_resumes_with_float32_tolerance(self):
        experiment = recipe("cuda")

        def interrupt(row):
            if row["step"] == 8:
                raise Interrupted

        with tempfile.TemporaryDirectory() as root:
            straight, resumed = Path(root) / "straight", Path(root) / "resumed"
            train(experiment, straight, CompiledStepper)
            with self.assertRaises(Interrupted):
                train(experiment, resumed, CompiledStepper, interrupt)
            train(experiment, resumed, CompiledStepper)
            # CUDA embedding-index reductions can round differently after capture.
            # Compare all numerical states, keeping sampler and accuracy exact.
            def compare(first, second, path=""):
                if torch.is_tensor(first):
                    torch.testing.assert_close(first, second, rtol=2e-6, atol=2e-7)
                elif isinstance(first, dict):
                    self.assertEqual(first.keys(), second.keys())
                    for key in first:
                        if key not in ("training_seconds", "diagnostic_seconds", "wall_seconds",
                                       "peak_cuda_allocated_bytes", "peak_cuda_reserved_bytes"):
                            compare(first[key], second[key], path + "/" + str(key))
                elif isinstance(first, (list, tuple)):
                    self.assertEqual(len(first), len(second))
                    for a, b in zip(first, second, strict=True):
                        compare(a, b, path)
                elif isinstance(first, float) and not path.endswith("accuracy"):
                    self.assertTrue(math.isclose(first, second, rel_tol=2e-6, abs_tol=2e-7), path)
                else:
                    self.assertEqual(first, second, path)

            first, second = RunDirectory(straight), RunDirectory(resumed)
            for stream in STREAMS:
                compare(first.records(stream), second.records(stream))
            compare(first.result(), second.result())
            compare(first.checkpoint("cpu"), second.checkpoint("cpu"))


if __name__ == "__main__":
    unittest.main()
