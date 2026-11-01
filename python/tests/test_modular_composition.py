"""Independent source equations, finite-field labels, isolation and continuation."""

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import torch
from torch.nn import functional as F

from lab.dsl import (Budget, Checkpoint, Eager, Evaluate, Experiment, ModularChain, ModularPairs,
                     Schedule, ScheduleFreeAdamW, Seeds, SourceTransformer)
from lab.infrastructure.benchmarks import build_task
from lab.infrastructure.engine.eager import EagerStepper
from lab.infrastructure.nn import build_model
from lab.infrastructure.store import RunDirectory
from test_engine import Interrupted
from test_graphs import train

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("source_equations", ROOT / "review/sequence_model_source_check.py")
equations = importlib.util.module_from_spec(spec)
spec.loader.exec_module(equations)


class ModularCompositionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_pinned_source_initialization_and_forward_fixture(self):
        fixture = json.loads((ROOT / "python/tests/fixtures/source_transformer.json").read_text())
        for case in fixture["cases"]:
            model = build_model(SourceTransformer(case["width"], case["depth"], case["heads"], case["length"]),
                                case["prime"], case["seed"])
            flat = torch.cat([p.detach().flatten() for p in model.parameters()])
            self.assertEqual(hashlib.sha256(flat.numpy().astype("<f4").tobytes()).hexdigest(), case["initial_sha256"])
            torch.testing.assert_close(model(torch.tensor(case["tokens"])), torch.tensor(case["logits"]),
                                       rtol=1e-6, atol=1e-7)
            self.assertIs(model.lm_head.weight, model.token_embedding_table.weight)

    def test_source_equations_and_all_gradients_in_double_precision(self):
        for width, depth, heads, length in ((8, 2, 2, 5), (16, 3, 2, 7), (8, 1, 1, 1)):
            model = build_model(SourceTransformer(width, depth, heads, length), 7, 11).double()
            tokens = torch.arange(2 * length).reshape(2, length) % 7
            targets = (tokens + 1) % 7
            actual, expected = model(tokens), equations.equations(model, tokens)
            torch.testing.assert_close(actual, expected, rtol=1e-8, atol=1e-10)
            gradients = [torch.autograd.grad(F.cross_entropy(value.flatten(0, 1), targets.flatten()),
                                            tuple(model.parameters())) for value in (actual, expected)]
            for a, b in zip(*gradients, strict=True):
                torch.testing.assert_close(a, b, rtol=1e-8, atol=1e-10)
            self.assertEqual(sum(p.numel() for p in model.parameters()),
                             7 * width + depth * (4 * width**2 + 3 * width * ((8 * width + 1) // 3)))

    def test_pairs_are_exhaustive_disjoint_and_independently_labeled(self):
        for prime in (2, 7, 17, 97, 193):
            for operation in ("mixed", "division"):
                task = build_task(ModularPairs(prime, operation), 11, torch.device("cpu"))
                rows = torch.cat(tuple(task.splits.values())).tolist()
                self.assertEqual(len(set(tuple(row[:2]) for row in rows)), prime * (prime - (operation == "division")))
                for a, b, label in rows:
                    if operation == "division":
                        self.assertEqual(label * b % prime, a)
                    else:
                        self.assertEqual(label, a * (a * a + b) % prime)
                again = build_task(ModularPairs(prime, operation), 11, torch.device("cpu"))
                self.assertEqual(task.summary(), again.summary())
        full = build_task(ModularPairs(7, train_fraction=1.), 11, torch.device("cpu"))
        self.assertEqual(len(full.splits["train"]), 49)
        self.assertEqual(full.spec.observed, ("train",))

    def test_chains_have_independent_horner_labels_and_final_only_longer_split(self):
        for prime, steps, train_count, heldout_count in ((3, 1, 14, 10), (17, 1, 2048, 1024), (97, 8, 50, 30)):
            task = build_task(ModularChain(prime, steps, train_count, heldout_count, 20), 11, torch.device("cpu"))
            seen = set()
            for name, rows in task.splits.items():
                for row in rows.tolist():
                    inputs, target = row[:-1], row[-1]
                    self.assertNotIn(tuple(inputs), seen)
                    seen.add(tuple(inputs))
                    value = inputs[0]
                    for b, c in zip(inputs[1::2], inputs[2::2], strict=True):
                        value = ((value * value + b) * value + c) % prime
                    self.assertEqual(value, target)
                self.assertEqual(rows.shape[1] - 1, 1 + (4 if name == "longer" else 2) * steps)
            self.assertEqual(task.spec.last, ("longer",))
            self.assertNotIn("longer", task.spec.observed)
            self.assertIsNone(task.spec.selection)

    def test_forward_does_not_read_answer_or_intermediates(self):
        task = build_task(ModularChain(7, 2, 8, 8, 8), 11, torch.device("cpu"))
        batch = task.splits["train"]
        captured = []

        def model(inputs):
            captured.append(inputs.clone())
            return torch.zeros(*inputs.shape, 7)

        logits, targets = task.forward(model, batch)
        self.assertEqual(logits.shape, (8, 7))
        torch.testing.assert_close(captured[0], batch[:, :-1])
        torch.testing.assert_close(targets, batch[:, -1])

    def test_copy_last_chain_baseline_counts_all_zero_mixed_pairs(self):
        for prime in (3, 7, 17):
            task = build_task(ModularChain(prime, 1, prime**3 - 1, 1, 1), 11, torch.device("cpu"))
            rows = torch.cat((task.splits["train"], task.splits["heldout"]))
            self.assertEqual(int((rows[:, -2] == rows[:, -1]).sum()), prime * (2 * prime - 1))
            self.assertEqual(task.summary()["full_domain_copy_last_probability"], (2 * prime - 1) / prime**2)

    def test_source_schedule_free_training_resumes_exactly_on_cpu(self):
        experiment = Experiment(model=SourceTransformer(8, 2, 2, 9),
            benchmark=ModularChain(7, 2, 16, 8, 8), optimizer=ScheduleFreeAdamW(lr=.003),
            schedule=Schedule(warmup=2), budget=Budget(8, 8), seeds=Seeds(11, 11, 11),
            evaluate=Evaluate(4, 8), checkpoint=Checkpoint(4), execution=Eager(device="cpu"))

        def interrupt(row):
            if row["step"] == 4:
                raise Interrupted

        with tempfile.TemporaryDirectory() as temporary:
            straight, resumed = Path(temporary) / "straight", Path(temporary) / "resumed"
            train(experiment, straight, EagerStepper)
            with self.assertRaises(Interrupted):
                train(experiment, resumed, EagerStepper, interrupt)
            train(experiment, resumed, EagerStepper)
            first, second = RunDirectory(straight).checkpoint("cpu"), RunDirectory(resumed).checkpoint("cpu")
            for name in first["model"]:
                torch.testing.assert_close(first["model"][name], second["model"][name], rtol=0, atol=0)
            for key in ("cursor", "rotary_random_state", "step", "examples_seen"):
                self.assertEqual(first[key], second[key])
            torch.testing.assert_close(first["permutation"], second["permutation"], rtol=0, atol=0)

    def test_invalid_resource_and_sampling_specs_are_rejected(self):
        for kwargs in ({"width": 7}, {"heads": 0}, {"depth": 0}, {"context": 0}, {"rope_base": float("nan")}):
            with self.assertRaises(ValueError):
                SourceTransformer(**kwargs)
        for kwargs in ({"prime": 4}, {"operation": "unknown"}, {"train_fraction": 0}, {"exponent": 2}):
            with self.assertRaises(ValueError):
                ModularPairs(**kwargs)
        with self.assertRaises(ValueError):
            ModularChain(2, 1, 5, 5, 4)
        with self.assertRaises(ValueError):
            build_model(SourceTransformer(8, 1, 2, 2), 7, 11)(torch.zeros(1, 3, dtype=torch.long))


if __name__ == "__main__":
    unittest.main()
