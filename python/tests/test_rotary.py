"""Browser fixtures, tied gradients, sampler continuation and a real lab run.

Independent fixture hashes were generated from the browser's PyTorch export
at grokking_abmod f49969393a7482490d45483d8a4fd3046b1adede. Tests do not
need that sibling checkout. The gradient checks include all three roles of E.
"""

import copy
from contextlib import redirect_stdout
import csv
import hashlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest

import torch
from torch.func import functional_call
from torch.nn import functional as F

from lab.application.study import Study, run_study
from lab.dsl import (AdamW, Budget, Checkpoint, DenseBranch, Eager, Evaluate, Experiment, LabDivision,
                     ReLU, Rotate90, Schedule, Seeds, SiLU, SwiGLU)
from lab.infrastructure.benchmarks import build_task
from lab.infrastructure.benchmarks.rotary import LabSampler, fingerprint
from lab.infrastructure.benchmarks.samplers import gather
from lab.infrastructure.engine import Engine
from lab.infrastructure.nn import build_model
from lab.infrastructure.store import RunDirectories

CPU = torch.device("cpu")


def digest(tensor, dtype):
    return hashlib.sha256(tensor.detach().cpu().numpy().astype(dtype).tobytes()).hexdigest()


class RotaryTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_browser_initial_checkpoint_and_paired_silu_weights(self):
        relu = build_model(Rotate90(128, DenseBranch(ReLU())), 97, 11)
        silu = build_model(Rotate90(128, DenseBranch(SiLU())), 97, 11)
        flat = torch.cat([p.detach().flatten() for p in relu.parameters()])
        self.assertEqual(digest(flat, "<f4"),
                         "b90448461caff3b6e04bd9ead0a3f7e7f055e8c1cd71ce19f1069186c74ae04b")
        for a, b in zip(relu.parameters(), silu.parameters(), strict=True):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertEqual(list(dict(relu.named_parameters())), ["E", "A"])
        self.assertEqual(sum(p.numel() for p in relu.parameters()), 28800)
        gated = build_model(Rotate90(128, SwiGLU(43)), 97, 11)
        self.assertEqual(sum(p.numel() for p in gated.parameters()), 28928)
        torch.testing.assert_close(relu.E, gated.E, rtol=0, atol=0)

    def test_browser_split_fixture_and_independent_modular_oracle(self):
        task = build_task(LabDivision(), 11, CPU)
        train, heldout = task.splits["train"], task.splits["heldout"]
        self.assertEqual(train.shape, (4656, 3))
        self.assertEqual(heldout.shape, (4656, 3))
        self.assertEqual(fingerprint(train),
                         "af1518d3d6d9f078a11a6cfcdd0ed6318c2386ea41a9b6d9826d4064e1c1b01e")
        self.assertEqual(fingerprint(heldout),
                         "fabbebaf7ab9a960308e775938abe42b8f5ced24da85f69f9890c078467c94ec")
        self.assertEqual(train[:4].tolist(), [[32, 48, 33], [68, 19, 24], [75, 60, 74], [49, 24, 95]])
        rows = torch.cat((train, heldout))
        self.assertEqual(len(set(map(tuple, rows[:, :2].tolist()))), 9312)
        self.assertTrue(bool((rows[:, 1] != 0).all()))
        self.assertTrue(bool(((rows[:, 2] * rows[:, 1] - rows[:, 0]) % 97 == 0).all()))
        self.assertEqual(int((rows[:, 0] == 0).sum()), 96)

    def test_browser_sampler_fixture_across_three_epochs(self):
        sampler = LabSampler(4656, 512, 11)
        batches = [gather(sampler.next()[0]) for _ in range(21)]
        self.assertEqual([len(b) for b in batches], [512] * 9 + [48] + [512] * 9 + [48, 512])
        self.assertEqual(digest(torch.cat(batches), "<i8"),
                         "245d3c3e5b08ad5aa96fe0307175fba4fd3af991d9ee99d925f1d2b0564f50b7")
        self.assertEqual(sampler.sizes(), [48, 512])

    def test_sampler_restore_preserves_rng_and_short_tail(self):
        sampler = LabSampler(4656, 512, 11)
        for _ in range(9):
            sampler.next()
        state = copy.deepcopy(sampler.state())
        expected = [gather(sampler.next()[0]) for _ in range(23)]
        restored = LabSampler(4656, 512, 99)
        restored.restore(state)
        for batch in expected:
            torch.testing.assert_close(gather(restored.next()[0]), batch, rtol=0, atol=0)

    def test_complete_cpu_protocol_matches_the_original_export(self):
        source = Path(__file__).resolve().parents[2] / "third_party/rotate90/reference.py"
        spec = importlib.util.spec_from_file_location("rotate90_reference", source)
        reference = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reference)
        stream = io.StringIO()
        with redirect_stdout(stream):
            original = reference.train(128, 11, 2, continue_after_target=True)
        expected = list(csv.DictReader(io.StringIO(stream.getvalue())))
        experiment = Experiment(model=Rotate90(128, DenseBranch(ReLU())), benchmark=LabDivision(),
                                optimizer=AdamW(lr=0.01, betas=(0.9, 0.98), eps=1e-8, weight_decay=1.0),
                                schedule=Schedule(warmup=10, inclusive=True), budget=Budget(updates=20, batch=512),
                                seeds=Seeds(model=11, data=11, batches=11), evaluate=Evaluate(every=10, batch=512),
                                execution=Eager(device="cpu"), checkpoint=Checkpoint(every=10))
        study = Study("rotary-reference", "Two full browser epochs on the current PyTorch", {"run": experiment})
        with tempfile.TemporaryDirectory() as directory:
            runs = RunDirectories(Path(directory))
            run_study(study, [], runs, Engine(), lambda label, row: None)
            run = runs.open("run")
            for actual, check in zip(run.records("history"), expected, strict=True):
                self.assertEqual(actual["step"], int(check["step"]))
                for split, prefix in (("train", "train"), ("heldout", "test")):
                    self.assertEqual(actual[split]["loss"], float(check[f"{prefix}_loss"]))
                    self.assertEqual(actual[split]["accuracy"], float(check[f"{prefix}_accuracy"]))
            for name, parameter in original.state_dict().items():
                torch.testing.assert_close(run.checkpoint(CPU)["model"][name], parameter, rtol=0, atol=0)

    def test_tied_embedding_gradient_sums_both_operands_and_readout(self):
        model = build_model(Rotate90(4, DenseBranch(ReLU())), 5, 11).double()
        inputs, targets = torch.tensor([[0, 1], [2, 3], [4, 2]]), torch.tensor([0, 4, 2])
        ours = model(inputs)
        F.cross_entropy(ours, targets).backward()
        ea, eb, er = [model.E.detach().clone().requires_grad_() for _ in range(3)]
        matrix = model.A.detach().clone().requires_grad_()
        rotation = torch.tensor([[0., 1., 0., 0.], [-1., 0., 0., 0.],
                                 [0., 0., 0., 1.], [0., 0., -1., 0.]], dtype=torch.float64)

        def normalize(x):
            return x / torch.sqrt(x.square().mean(dim=-1, keepdim=True) + 1e-5)

        x = ea[inputs[:, 0]] + eb[inputs[:, 1]] @ rotation
        independent = normalize(x + torch.relu(normalize(x) @ matrix)) @ er.T
        gradients = torch.autograd.grad(F.cross_entropy(independent, targets), (ea, eb, er, matrix))
        torch.testing.assert_close(ours, independent, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(model.E.grad, sum(gradients[:3]), rtol=1e-11, atol=1e-11)
        torch.testing.assert_close(model.A.grad, gradients[3], rtol=1e-11, atol=1e-11)

    def test_every_parameter_has_the_finite_difference_gradient(self):
        inputs = torch.tensor([[0, 1], [2, 3], [4, 2]])
        for branch in (DenseBranch(ReLU()), DenseBranch(SiLU()), SwiGLU(3)):
            with self.subTest(branch=branch):
                model = build_model(Rotate90(4, branch), 5, 11).double()
                names, parameters = zip(*model.named_parameters(), strict=True)

                def evaluate(*values):
                    return functional_call(model, dict(zip(names, values, strict=True)), (inputs,))

                self.assertTrue(torch.autograd.gradcheck(evaluate, parameters))
                model(inputs).square().sum().backward()
                for p in parameters:
                    self.assertIsNotNone(p.grad)
                    self.assertGreater(float(p.grad.norm()), 0)

    def test_real_training_uses_the_pair_protocol_and_records_norms(self):
        experiment = Experiment(model=Rotate90(8, DenseBranch(SiLU())),
                                benchmark=LabDivision(prime=5),
                                optimizer=AdamW(lr=0.01, betas=(0.9, 0.98), weight_decay=0.0),
                                schedule=Schedule(warmup=10, inclusive=True), budget=Budget(updates=6, batch=4),
                                seeds=Seeds(model=11, data=11, batches=11), evaluate=Evaluate(every=3, batch=4),
                                execution=Eager(device="cpu"), checkpoint=Checkpoint(every=3))
        study = Study("rotary", "Rotate90 protocol smoke run", {"run": experiment})
        with tempfile.TemporaryDirectory() as root:
            runs = RunDirectories(Path(root))
            result = run_study(study, [], runs, Engine(), lambda label, row: None)["run"]
            history = runs.open("run").records("history")
            self.assertEqual(result["stop"]["reason"], "budget")
            self.assertEqual([point["step"] for point in history], [0, 3, 6])
            self.assertEqual(history[-1]["train"]["examples"], 10)
            self.assertEqual(history[-1]["heldout"]["examples"], 10)
            self.assertIn("parameter_norm", history[-1])
            self.assertLess(history[-1]["train"]["loss"], history[0]["train"]["loss"])

    def test_invalid_model_specs_fail_early(self):
        for width in (0, 1, 3, 4.5):
            with self.assertRaises(ValueError):
                Rotate90(width, DenseBranch(ReLU()))
        with self.assertRaises(ValueError):
            SwiGLU(0)
        with self.assertRaises(ValueError):
            Rotate90(4, DenseBranch(ReLU()), eps=0)
