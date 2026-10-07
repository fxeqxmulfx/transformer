"""Counterexamples and noninterference for all six offline measurements."""

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
from probes.ablation import remove_head, remove_subspace
from probes.capture import collect, forward
from probes.features import fit_linear, linear_probe, linear_scores, neurons, spectrum
from probes.gradients import agreement
from probes.suite import Suite
from probes.updates import decompose


def experiment(device="cpu"):
    base = dict(load(STUDY).select([]))["gptmini-seed1"]
    for path, value in (("model.width", 16), ("model.block.attention.heads", 2),
                        ("benchmark.prime", 5), ("execution.device", device)):
        base = swap(base, path, value)
    return base


class MeasurementTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_frozen_probe_generalizes_a_linear_code(self):
        train = torch.tensor([[-2., 0.], [-1., 0.], [1., 0.], [2., 0.]])
        labels = torch.tensor([0, 0, 1, 1])
        fit = fit_linear(train, labels, 2)
        self.assertEqual(linear_scores(fit, torch.tensor([[-3., 0.], [3., 0.]])).argmax(-1).tolist(), [0, 1])

    def test_heldout_labels_do_not_fit_or_select_the_probe(self):
        x = torch.tensor([[-2.], [-1.], [1.], [2.], [-3.], [3.]])
        labels = torch.tensor([0, 0, 1, 1, 0, 1])
        train, heldout = torch.arange(4), torch.arange(4, 6)
        good = linear_probe(x, labels, train, heldout, 2)
        labels[heldout] = labels[heldout].flip(0)
        wrong = linear_probe(x, labels, train, heldout, 2)
        self.assertEqual(good["train_accuracy"], wrong["train_accuracy"])
        self.assertEqual(good["heldout_accuracy"], 1)
        self.assertEqual(wrong["heldout_accuracy"], 0)

    def test_training_only_memorization_does_not_supply_heldout_features(self):
        x = torch.cat((torch.eye(4), torch.zeros(4, 4)))
        labels = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
        result = linear_probe(x, labels, torch.arange(4), torch.arange(4, 8), 2)
        self.assertEqual(result["train_accuracy"], 1)
        self.assertEqual(result["heldout_accuracy"], .5)

    def test_spectrum_has_known_rank_and_is_rotation_scale_invariant(self):
        x = torch.tensor([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]])
        original = spectrum(x)[0]
        rotation = torch.tensor([[.6, -.8], [.8, .6]])
        changed = spectrum(7 * x @ rotation + 5)[0]
        self.assertAlmostEqual(original["energy_entropy_rank"], 2)
        self.assertAlmostEqual(original["stable_rank"], 2)
        self.assertEqual(original["rank99"], 2)
        self.assertAlmostEqual(original["energy_entropy_rank"], changed["energy_entropy_rank"], places=10)

    def test_constant_features_do_not_claim_a_low_dimensional_rule(self):
        self.assertIsNone(spectrum(torch.ones(8, 3))[0]["energy_entropy_rank"])

    def test_neuron_profiles_transfer_and_exclude_zero_quotient(self):
        labels = torch.tensor([0, 0, 1, 1, 2, 2] * 2)
        x = torch.tensor([[100., 100.], [100., 100.], [2., 0.], [2., 0.], [0., 2.], [0., 2.]] * 2)
        result = neurons(x, labels, torch.arange(6), torch.arange(6, 12), 3)
        self.assertTrue(result["class_coverage"])
        self.assertEqual(result["exact_zero_train_fraction"], .5)
        self.assertAlmostEqual(result["median_profile_cosine"], 1)
        self.assertEqual(result["median_train_selectivity"], 1)

    def test_gradient_agreement_distinguishes_alignment_conflict_and_zero(self):
        self.assertEqual(agreement(torch.tensor([[1., 0.], [2., 0.]]))["mean_pair_cosine"], 1)
        self.assertEqual(agreement(torch.tensor([[1., 0.], [-2., 0.], [0., 0.]]))["mean_pair_cosine"], -1)
        self.assertIsNone(agreement(torch.zeros(2, 3))["mean_pair_cosine"])

    def test_confidence_growth_is_not_decision_geometry_change(self):
        x = torch.tensor([[2., 0.], [0., 2.]])
        result = decompose(x, 3 * x + torch.tensor([[7.], [-4.]]), torch.tensor([0, 1]))
        self.assertAlmostEqual(result["fitted_global_scale"], 3)
        self.assertAlmostEqual(result["scale_change_energy_fraction"], 1)
        self.assertAlmostEqual(result["geometry_change_energy_fraction"], 0)
        self.assertEqual(result["prediction_change_fraction"], 0)
        self.assertLess(result["after_loss"], result["before_loss"])

    def test_decision_changes_are_detected(self):
        x = torch.tensor([[2., -1., -1.], [-1., 2., -1.], [-1., -1., 2.]])
        result = decompose(x, x.roll(1, dims=1), torch.arange(3))
        self.assertEqual(result["prediction_change_fraction"], 1)
        self.assertAlmostEqual(result["geometry_change_energy_fraction"], .25)
        self.assertLess(result["fitted_global_scale"], 0)
        self.assertFalse(result["global_scaling_preserves_decisions"])
        self.assertLess(result["energy_decomposition_error"], 1e-12)

    def test_softmax_invisible_shifts_produce_no_update_evidence(self):
        x = torch.tensor([[2., 0.], [0., 2.]])
        result = decompose(x, x + 8, torch.tensor([0, 1]))
        self.assertIsNone(result["scale_change_energy_fraction"])
        self.assertEqual(result["change_rms"], 0)


class InterventionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.experiment = experiment()
        self.task = ModularTask(self.experiment.benchmark, self.experiment.seeds.data, torch.device("cpu"))
        self.model = build_model(self.experiment.model, self.task.vocab, 1)
        self.suite = Suite(self.experiment, self.task)

    def test_head_ablation_matches_removing_projection_columns(self):
        other = copy.deepcopy(self.model)
        with torch.no_grad():
            other.blocks[0].attention.output.weight[:, :8] = 0
        with remove_head(self.model.blocks[0].attention, 0, 2):
            actual = forward(self.model, self.suite.orbits.inputs)
        expected = forward(other, self.suite.orbits.inputs)
        torch.testing.assert_close(actual, expected)

    def test_subspace_ablation_preserves_the_mean_and_other_positions(self):
        seen = []
        hook = self.model.blocks[0].register_forward_hook(lambda module, inputs, output: seen.append(output.detach().clone()))
        forward(self.model, self.suite.orbits.inputs)
        original = seen[-1]
        mean = original[:, 4].double().mean(0)
        with remove_subspace(self.model.blocks[0], mean, torch.eye(16, dtype=torch.float64)):
            forward(self.model, self.suite.orbits.inputs)
            # The earlier capture hook reads the pre-intervention tensor.
            final = []
            later = self.model.blocks[0].register_forward_hook(lambda module, inputs, output: final.append(output.detach().clone()))
            forward(self.model, self.suite.orbits.inputs)
            later.remove()
        hook.remove()
        torch.testing.assert_close(final[-1][:, :4], original[:, :4])
        torch.testing.assert_close(final[-1][:, 4].double(), mean.expand(len(original), -1), atol=1e-8, rtol=1e-6)

    def test_intervention_hooks_are_removed_after_failure(self):
        attention = self.model.blocks[0].attention
        with self.assertRaisesRegex(RuntimeError, "probe failure"):
            with remove_head(attention, 0, 2):
                raise RuntimeError("probe failure")
        self.assertFalse(attention.output._forward_pre_hooks)
        with self.assertRaisesRegex(RuntimeError, "probe failure"):
            with remove_subspace(self.model.blocks[0], torch.zeros(16), torch.eye(16)[:, :8]):
                raise RuntimeError("probe failure")
        self.assertFalse(self.model.blocks[0]._forward_hooks)

    def test_capture_restores_hooks_modes_and_rng_after_failed_forward(self):
        self.model.train()
        self.model.blocks[0].eval()
        modes = [module.training for module in self.model.modules()]
        rng = torch.random.get_rng_state().clone()

        def fail(*args):
            torch.rand(1)
            raise RuntimeError("readout failure")

        with patch.object(self.model.readout, "forward", fail), self.assertRaisesRegex(RuntimeError, "readout failure"):
            collect(self.model, self.suite.orbits.inputs)
        self.assertEqual(modes, [module.training for module in self.model.modules()])
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        self.assertFalse(any(module._forward_hooks or module._forward_pre_hooks for module in self.model.modules()))

    def verify_noninterference(self, device):
        spec = experiment(device)
        task = ModularTask(spec.benchmark, spec.seeds.data, torch.device(device))
        model = build_model(spec.model, task.vocab, 1).to(device)
        other = copy.deepcopy(model)
        optimizers = [torch.optim.AdamW(m.parameters(), lr=.001, betas=(.9, .98)) for m in (model, other)]
        data = task.splits["train"]

        def update(m, optimizer):
            optimizer.zero_grad()
            output, target = task.forward(m, data)
            task.loss(output, target).backward()
            optimizer.step()

        for m, optimizer in zip((model, other), optimizers):
            update(m, optimizer)
        before = copy.deepcopy(model.state_dict())
        previous_gradients = [p.grad.clone() for p in model.parameters()]
        previous_rng = torch.random.get_rng_state().clone()
        cuda_rng = torch.cuda.get_rng_state().clone() if device == "cuda" else None
        result = Suite(spec, task).observe(model, 1)
        self.assertEqual(result["feature_position"], "equals_in_raw_five_token_prefix")
        self.assertEqual(len(result["ablations"]["heads"]), 4)
        self.assertEqual(len(result["ablations"]["subspaces"]), 6)
        for key, tensor in before.items():
            self.assertTrue(torch.equal(tensor, model.state_dict()[key]), key)
        for gradient, p in zip(previous_gradients, model.parameters()):
            self.assertTrue(torch.equal(gradient, p.grad))
        self.assertTrue(torch.equal(previous_rng, torch.random.get_rng_state()))
        if cuda_rng is not None:
            self.assertTrue(torch.equal(cuda_rng, torch.cuda.get_rng_state()))
        for m, optimizer in zip((model, other), optimizers):
            update(m, optimizer)
        for a, b in zip(model.parameters(), other.parameters()):
            self.assertTrue(torch.equal(a, b))
        for a, b in zip(optimizers[0].state.values(), optimizers[1].state.values()):
            for key in a:
                self.assertTrue(torch.equal(a[key], b[key]))

    def test_actual_adamw_updates_are_unchanged_on_cpu(self):
        self.verify_noninterference("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_actual_adamw_updates_are_unchanged_on_cuda(self):
        self.verify_noninterference("cuda")


if __name__ == "__main__":
    unittest.main()
