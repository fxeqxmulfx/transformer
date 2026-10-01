"""The baseline uses the raw AMSGradW recurrence and benchmark initialization."""

import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

from experiments.optimizer_benchmark.attention import make_model
from experiments.synthetic_trainers.config import ModelSpec, TrainConfig
from experiments.synthetic_trainers.runtime import optimizer_description, optimizer_for
from experiments.synthetic_trainers.training import train_run
from experiments.synthetic_trainers.specs import TaskSpec


class BaselineOptimizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_raw_moments_decoupled_decay_and_retained_maximum_match_scalar_recurrence(self):
        model = nn.Linear(1, 1, bias=False).double()
        model.weight.data.fill_(2.0)
        config = TrainConfig(optimizer="amsgradw", learning_rate=.01, weight_decay=.2,
                             beta1=.9, beta2=.5, optimizer_epsilon=.1, grad_clip=None)
        optimizer = optimizer_for(model, config)
        value, first, second, maximum = 2.0, 0.0, 0.0, 0.0
        for gradient in (2.0, -1.0, .25, 0.0):
            model.weight.grad = torch.full_like(model.weight, gradient)
            first = .9 * first + .1 * gradient
            second = .5 * second + .5 * gradient ** 2
            maximum = max(maximum, second)
            value = (1 - .01 * .2) * value - .01 * first / (.1 + math.sqrt(maximum))
            optimizer.step()
            self.assertAlmostEqual(model.weight.item(), value, places=12)
            self.assertAlmostEqual(optimizer.state[model.weight]["m"].item(), first, places=12)
            self.assertAlmostEqual(optimizer.state[model.weight]["v"].item(), second, places=12)
            self.assertAlmostEqual(optimizer.state[model.weight]["maximum"].item(), maximum, places=12)
        self.assertFalse(optimizer_description(config)["bias_correction"])
        self.assertFalse(optimizer.guarded)

    def test_decay_does_not_enter_moments_and_tied_parameters_are_updated_once(self):
        class Tied(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(2, 2))
                self.alias = self.weight
                self.scalar = nn.Parameter(torch.ones(1))
        model = Tied()
        config = TrainConfig(optimizer="amsgradw", learning_rate=.1, weight_decay=.2)
        optimizer = optimizer_for(model, config)
        for parameter in model.parameters():
            parameter.grad = torch.zeros_like(parameter)
        optimizer.step()
        for parameter in model.parameters():
            self.assertTrue(torch.allclose(parameter, torch.full_like(parameter, .98)))
            self.assertEqual(float(optimizer.state[parameter]["m"].abs().sum()), 0)
            self.assertEqual(float(optimizer.state[parameter]["maximum"].abs().sum()), 0)
        self.assertEqual(len(optimizer.param_groups[0]["params"]), 2)

    def test_saved_initial_state_matches_existing_softmax_optimizer_benchmark(self):
        spec = TaskSpec(task="copy", length=4, symbols=8)
        model_spec = ModelSpec(width=8, layers=1, heads=1, init_std=.02)
        config = TrainConfig(optimizer="amsgradw", grad_clip=None, steps=1, eval_every=1,
                             train_examples=2, validation_examples=2, test_examples=2, seed=7)
        metrics = {"sequence_accuracy": 1, "balanced_accuracy": 1, "loss": 0, "token_accuracy": 1}
        with tempfile.TemporaryDirectory() as root:
            with patch("experiments.synthetic_trainers.training.evaluate", return_value=metrics):
                report = train_run(spec, model_spec, config, root)
            expected = make_model(model_spec.reference_config(spec.vocab_size, spec.context_length), "softmax", 7)
            saved = torch.load(Path(root) / "best.pt", weights_only=True)
            self.assertTrue(all(torch.equal(value, saved[name]) for name, value in expected.state_dict().items()))
            self.assertEqual(report["provenance"]["optimizer"]["name"], "amsgradw")
            self.assertTrue(any(name.endswith("coordinate.py") for name in report["provenance"]["source_hashes"]))

    def test_amsgradw_completes_a_real_study_with_unclipped_gradients(self):
        config = TrainConfig(optimizer="amsgradw", grad_clip=None, study="double_descent", steps=2, eval_every=1,
                             train_examples=3, validation_examples=2, test_examples=2, eval_lengths=(8,))
        with tempfile.TemporaryDirectory() as root:
            report = train_run(TaskSpec(task="copy", length=4, symbols=8), ModelSpec(width=8, layers=1, heads=1, init_std=.02), config, root)
            self.assertEqual(report["steps_completed"], 2)
            self.assertTrue(math.isfinite(report["test_final"]["in_distribution"]["loss"]))
            self.assertIsNone(report["provenance"]["optimizer"]["gradient_clip"])

    def test_invalid_optimizer_and_initialization_settings_are_rejected(self):
        for fields in ({"optimizer": "amsgrad"}, {"beta1": 1}, {"beta2": -1}, {"optimizer_epsilon": 0}):
            with self.assertRaises(ValueError):
                TrainConfig(**fields)
        with self.assertRaises(ValueError):
            ModelSpec(init_std=float("nan"))
