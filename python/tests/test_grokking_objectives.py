"""Shared-supervision counterexamples and unchanged native AdamW updates."""

import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch

from lab.domain.spec import swap
from lab.infrastructure.benchmarks.modular import ModularTask
from lab.infrastructure.loader import load
from lab.infrastructure.nn import build_model

STUDY = Path(__file__).resolve().parents[2] / "experiments/grokking_internals"
sys.path.insert(0, str(STUDY))
from objective_components import energy_terms, measure, split_dot_terms
from probes.gradients import agreement


class ObjectiveTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_shared_target_can_align_opposing_answer_gradients(self):
        answer = torch.tensor([[1., 0.], [-1., 0.]], dtype=torch.float64)
        eos = torch.tensor([[0., 100.], [0., 100.]], dtype=torch.float64)
        full = (answer + eos) / 2
        self.assertEqual(agreement(answer)["mean_pair_cosine"], -1)
        self.assertGreater(agreement(full)["mean_pair_cosine"], .999)
        energies = energy_terms(answer, eos, full)
        self.assertEqual(energies["half_answer_EOS_dot"], 0)
        self.assertEqual(energies["energy_identity_absolute_error"], 0)
        self.assertEqual(energies["max_gradient_reconstruction_l2"], 0)

    def test_signed_cross_term_retains_complete_cancellation(self):
        answer = torch.tensor([[2., 0.], [0., 4.]])
        result = energy_terms(answer, -answer, torch.zeros_like(answer))
        self.assertEqual(result["full_squared_norm"], 0)
        self.assertEqual(result["half_answer_EOS_dot"], -5)
        self.assertEqual(result["energy_identity_absolute_error"], 0)

    def test_split_cosine_keeps_cross_component_terms(self):
        train = {"answer": torch.tensor([[2., 0.]]), "EOS": torch.tensor([[0., 2.]])}
        heldout = {"answer": torch.tensor([[0., 2.]]), "EOS": torch.tensor([[2., 0.]])}
        for split in (train, heldout):
            split["full"] = (split["answer"] + split["EOS"]) / 2
        result = split_dot_terms(train, heldout)
        self.assertEqual(result["quarter_component_dots"]["answer_answer"], 0)
        self.assertEqual(result["quarter_component_dots"]["answer_EOS"], 1)
        self.assertAlmostEqual(sum(result["terms_over_full_norm_product"].values()), 1)
        self.assertEqual(result["dot_identity_absolute_error"], 0)

    def build(self, device):
        spec = dict(load(STUDY).select([]))["gptmini-seed1"]
        for path, value in (("model.width", 16), ("model.block.attention.heads", 2), ("benchmark.prime", 5)):
            spec = swap(spec, path, value)
        task = ModularTask(spec.benchmark, spec.seeds.data, torch.device(device))
        return task, build_model(spec.model, task.vocab, 1).to(device)

    def test_actual_mask_and_gradients_reconstruct_with_unused_and_tied_parameters(self):
        task, model = self.build("cpu")
        model.register_parameter("unused", torch.nn.Parameter(torch.ones(3)))
        result = measure(model, task, batches=2, batch=3)
        self.assertIs(model.embed.weight, model.readout.weight)
        self.assertEqual(result["unique_trainable_tensors"], len(list(model.parameters())))
        self.assertEqual(len(list(model.named_parameters(remove_duplicate=False))),
                         result["unique_trainable_tensors"] + 1)
        for split in ("train", "heldout"):
            losses = result["loss_by_split"][split]
            self.assertAlmostEqual(losses["full"], (losses["answer"] + losses["EOS"]) / 2, places=6)
            errors = result["groups"]["all"]["energy_by_split"][split]
            self.assertLess(errors["max_gradient_reconstruction_relative_error"], 1e-6)
        self.assertIsNone(model.unused.grad)

    def test_mask_validation_restores_modes_after_failure(self):
        task, model = self.build("cpu")
        model.train()
        model.blocks[0].eval()
        modes = [m.training for m in model.modules()]
        original = task.forward
        with patch.object(task, "forward", side_effect=lambda m, rows:
                          (original(m, rows)[0][:, :1], original(m, rows)[1][:, :1])):
            with self.assertRaisesRegex(ValueError, "two supervised positions"):
                measure(model, task, batches=1, batch=2)
        self.assertEqual(modes, [m.training for m in model.modules()])

    def verify_updates(self, device):
        task, model = self.build(device)
        other = copy.deepcopy(model)
        optimizers = [torch.optim.AdamW(m.parameters(), lr=.001, betas=(.9, .98)) for m in (model, other)]

        def update(m, optimizer):
            optimizer.zero_grad()
            output, target = task.forward(m, task.splits["train"])
            task.loss(output, target).backward()
            optimizer.step()

        for m, optimizer in zip((model, other), optimizers):
            update(m, optimizer)
        before = copy.deepcopy(model.state_dict())
        gradients = [p.grad.clone() for p in model.parameters()]
        model.train()
        model.blocks[0].eval()
        modes = [m.training for m in model.modules()]
        rng = torch.random.get_rng_state().clone()
        cuda_rng = torch.cuda.get_rng_state().clone() if device == "cuda" else None
        measure(model, task, batches=2, batch=3)
        self.assertEqual(modes, [m.training for m in model.modules()])
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        if cuda_rng is not None:
            self.assertTrue(torch.equal(cuda_rng, torch.cuda.get_rng_state()))
        for key, tensor in before.items():
            self.assertTrue(torch.equal(tensor, model.state_dict()[key]), key)
        for saved, p in zip(gradients, model.parameters()):
            self.assertTrue(torch.equal(saved, p.grad))
        for m, optimizer in zip((model, other), optimizers):
            update(m, optimizer)
        for a, b in zip(model.parameters(), other.parameters()):
            self.assertTrue(torch.equal(a, b))
        for a, b in zip(optimizers[0].state.values(), optimizers[1].state.values()):
            for key in a:
                self.assertTrue(torch.equal(a[key], b[key]))

    def test_actual_adamw_updates_are_unchanged_on_cpu(self):
        self.verify_updates("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_actual_adamw_updates_are_unchanged_on_cuda(self):
        self.verify_updates("cuda")


if __name__ == "__main__":
    unittest.main()
