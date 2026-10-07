"""Functional projection and unchanged actual softmax/AdamW trajectories."""

import math
from pathlib import Path
import tempfile
import unittest

import torch
from torch.nn import functional as F

from lab.domain.grokking import GrokkingDiagnostics
from lab.domain.spec import swap
from lab.domain.training import Checkpoint, Eager
from lab.infrastructure.benchmarks.modular import ModularTask
from lab.infrastructure.engine.eager import EagerStepper
from lab.infrastructure.engine.grokking import GrokkingObserver, statistics
from lab.infrastructure.engine.grokking_internal import hidden_energy
from lab.infrastructure.nn import build_model
from lab.infrastructure.store import RunDirectory

from examples import gptmini, modular
from test_engine import Interrupted
from test_graphs import summary, train


def measure(logits, train=None, heldout=None):
    p, orbit, _ = logits.shape
    indices = torch.arange(p * orbit)
    targets = torch.arange(p).repeat_interleave(orbit)
    return statistics(logits, indices if train is None else train,
                      indices if heldout is None else heldout, targets)


def specimen(device="cpu"):
    base = modular(gptmini(16, 2, 2), prime=11, updates=12, batch=8, every=4,
                   execution=Eager(device=device, threads=1))
    return swap(base, "checkpoint", Checkpoint(every=4))


class GrokkingObserverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_projection_has_known_energy_and_rejects_bias_only_evidence(self):
        a, b = torch.tensor([1., -1., 0.]), torch.tensor([0., 1., -1.])
        x = torch.stack([torch.zeros(2, 3), torch.stack([a + b, a - b]),
                         torch.stack([-a + b, -a - b])])
        found = measure(x)
        self.assertAlmostEqual(found["invariant_energy_fraction"], .5)
        self.assertAlmostEqual(found["heldout_invariant_energy_fraction"], .5)
        self.assertLess(found["energy_decomposition_error"], 1e-12)
        self.assertLess(found["heldout_energy_decomposition_error"], 1e-12)
        shifted = x.double() * 3 + torch.tensor([11., -7., 2.])
        shifted += torch.arange(6).reshape(3, 2, 1)
        shifted[0] += 1000
        self.assertAlmostEqual(measure(shifted)["heldout_invariant_energy_fraction"], .5)
        constant = torch.tensor([11., -7., 2.]).expand(3, 2, 3)
        self.assertIsNone(measure(constant)["heldout_invariant_energy_fraction"])
        self.assertTrue(measure(constant)["constant_logits"])

    def test_training_memorization_cannot_supply_heldout_structure(self):
        x = torch.zeros(5, 4, 5)
        for q in range(5):
            x[q, :2, q] = 12.
        train_indices = torch.tensor([q * 4 + d for q in range(5) for d in (0, 1)])
        heldout = torch.tensor([q * 4 + d for q in range(5) for d in (2, 3)])
        found = measure(x, train_indices, heldout)
        self.assertLess(found["restricted_heldout_loss"], .01)
        self.assertAlmostEqual(found["heldout_restricted_loss"], math.log(5))
        self.assertIsNone(found["heldout_invariant_energy_fraction"])
        altered = x.clone()
        altered[:, :2] *= 100
        changed = measure(altered, train_indices, heldout)
        for key in ("heldout_invariant_energy_fraction", "heldout_residual_energy", "heldout_restricted_loss"):
            self.assertEqual(found[key], changed[key])

    def test_invariance_does_not_imply_correctness_and_singletons_are_rejected(self):
        x = torch.zeros(5, 4, 5)
        for q in range(5):
            x[q, :, (q + 1) % 5] = 10.
        found = measure(x)
        self.assertAlmostEqual(found["heldout_invariant_energy_fraction"], 1.)
        self.assertEqual(found["heldout_restricted_accuracy"], 0.)
        self.assertIsNone(measure(x, heldout=torch.tensor([0, 4, 8, 12, 16]))["heldout_invariant_energy_fraction"])
        with self.assertRaises(FloatingPointError):
            measure(x * float("nan"))
        with self.assertRaises(ValueError):
            measure(torch.zeros(4, 3, 4))

    def test_probe_preserves_rng_gradients_modes_and_uses_no_answer_token(self):
        base = specimen()
        task = ModularTask(base.benchmark, base.seeds.data, torch.device("cpu"))
        model = build_model(base.model, task.vocab, base.seeds.model)
        model.train()
        list(model.modules())[2].eval()
        for parameter in model.parameters():
            parameter.grad = torch.ones_like(parameter)
        modes = [module.training for module in model.modules()]
        before = summary(model.state_dict())
        gradients = summary([p.grad for p in model.parameters()])
        rng = torch.get_rng_state().clone()
        lengths = []

        def inspect(module, inputs):
            lengths.append(inputs[0].shape[1])
            torch.rand(1)

        handle = model.register_forward_pre_hook(inspect)
        observer = GrokkingObserver(GrokkingDiagnostics(batch=7, orbit_every=4), task)
        found = observer.observe(model, 4)
        handle.remove()
        inner = found["internal"]
        self.assertEqual(set(inner["features"]), {f"blocks.{i}{part}" for i in (0, 1)
                                                for part in ("", ".attention", ".ffn")})
        self.assertEqual(len(inner["attention"]), 2)
        for entry in inner["features"].values():
            self.assertLess(entry["energy_decomposition_error"], 1e-10)
        for entry in inner["attention"].values():
            self.assertEqual(len(entry["entropy_by_head"]), 2)
            for weights in entry["mean_weights_by_head_and_prompt_token"]:
                self.assertAlmostEqual(sum(weights), 1., places=6)
        self.assertTrue(all(not module._forward_hooks for module in model.modules()))
        self.assertEqual(set(lengths), {5})
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(modes, [module.training for module in model.modules()])
        self.assertEqual(before, summary(model.state_dict()))
        self.assertEqual(gradients, summary([p.grad for p in model.parameters()]))
        with torch.no_grad():
            logits, target = task.forward(model, task.splits["heldout"])
            expected = float(F.cross_entropy(logits[:, 0].double(), target[:, 0]))
        self.assertAlmostEqual(found["raw_heldout_loss"], expected, places=6)

    def test_hidden_energy_and_hooks_are_cleaned_up_after_forward_failure(self):
        x = torch.tensor([[[0., 0.], [0., 0.]], [[2., 0.], [0., 2.]],
                          [[0., -2.], [-2., 0.]]])
        energy = hidden_energy(x.reshape(6, 2), torch.arange(6), 3, 1e-12)
        self.assertAlmostEqual(energy["invariant_energy_fraction"], .5)
        base = specimen()
        task = ModularTask(base.benchmark, 0, torch.device("cpu"))
        model = build_model(base.model, task.vocab, 0)
        model.train()
        rng = torch.get_rng_state().clone()

        def fail(module, inputs):
            torch.rand(1)
            raise RuntimeError("forced probe failure")

        handle = model.register_forward_pre_hook(fail)
        observer = GrokkingObserver(GrokkingDiagnostics(batch=7, orbit_every=4), task)
        with self.assertRaisesRegex(RuntimeError, "forced probe failure"):
            observer.observe(model, 0)
        handle.remove()
        self.assertTrue(model.training)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(all(not module._forward_hooks for module in model.modules()))

    def check_trajectory(self, device):
        base = specimen(device)
        observed = swap(base, "diagnostics", GrokkingDiagnostics(orbit_every=4, batch=7))
        with tempfile.TemporaryDirectory() as root:
            plain, extra, resumed = [Path(root) / name for name in ("plain", "extra", "resumed")]
            train(base, plain, EagerStepper)
            train(observed, extra, EagerStepper)

            def interrupt(row):
                if row["step"] == 8:
                    raise Interrupted()

            with self.assertRaises(Interrupted):
                train(observed, resumed, EagerStepper, interrupt)
            train(observed, resumed, EagerStepper)
            a, b, c = [RunDirectory(path) for path in (plain, extra, resumed)]
            self.assertEqual(summary(a.checkpoint("cpu")), summary(b.checkpoint("cpu")))
            self.assertEqual(summary(b.checkpoint("cpu")), summary(c.checkpoint("cpu")))
            self.assertEqual(summary(b.records("history")), summary(c.records("history")))
            stripped = [{key: value for key, value in row.items() if key != "grokking"}
                        for row in b.records("history")]
            self.assertEqual(summary(a.records("history")), summary(stripped))

    def test_observation_and_resume_leave_actual_cpu_adamw_training_unchanged(self):
        self.check_trajectory("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_observation_and_resume_leave_actual_cuda_adamw_training_unchanged(self):
        self.check_trajectory("cuda")
