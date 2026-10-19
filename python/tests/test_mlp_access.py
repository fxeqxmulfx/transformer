"""Independent derivatives and finite differences for the offline M9.4 diagnostic."""

import importlib.util
from pathlib import Path
import unittest

import torch
from torch.func import functional_call

from lab.dsl import MLP, OneHot, Scalar, SiLU, TorchDefault
from lab.infrastructure.nn import build_model

SOURCE = Path(__file__).resolve().parents[2] / "experiments/mod97_representation_controls/access.py"
SPEC = importlib.util.spec_from_file_location("mlp_access", SOURCE)
access = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(access)


class MLPAccessTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.pairs = torch.tensor([(a, b) for a in range(5) for b in range(1, 5)])
        labels = torch.tensor([a * pow(b, -1, 5) % 5 for a, b in self.pairs.tolist()])
        self.targets = (labels.double() / 2 - 1).reshape(5, 4)

    def models(self):
        for encoding in (Scalar(), OneHot()):
            yield build_model(MLP(width=4, depth=2, encoding=encoding, activation=SiLU(),
                                  init=TorchDefault(), context=2), 5, 11).double()

    def test_batched_jacobian_matches_every_actual_parameter_derivative(self):
        for model in self.models():
            expected = []
            for pair in self.pairs:
                derivatives = torch.autograd.grad(model(pair[None]).squeeze(), tuple(model.parameters()))
                expected.append(torch.cat([value.flatten() for value in derivatives]))
            torch.testing.assert_close(access.prediction_jacobian(model, self.pairs, chunk=3),
                                       torch.stack(expected), rtol=1e-13, atol=1e-15)

    def test_centered_target_contraction_matches_direct_autograd(self):
        for model in self.models():
            jacobian = access.prediction_jacobian(model, self.pairs, chunk=7)
            before = {name: value.detach().clone() for name, value in model.named_parameters()}
            predictions = model(self.pairs).reshape_as(self.targets)
            centered_target = access.interaction(self.targets)
            direct = torch.autograd.grad((predictions * centered_target).sum(), tuple(model.parameters()))
            expected = sum(float(value.square().sum()) for value in direct)
            result = access.summarize(jacobian, predictions.detach(), self.targets, model)
            self.assertAlmostEqual(result["target"]["jacobian_target_contraction_squared"], expected, places=14)
            self.assertAlmostEqual(result["interaction_jacobian_energy"],
                                   sum(row["interaction_jacobian_energy"] for row in result["layers"].values()),
                                   places=14)
            self.assertLess(result["layers"]["output.bias"]["interaction_jacobian_energy"], 1e-25)
            self.assertGreater(result["target"]["trace_normalized_alignment"], 0)
            for name, parameter in model.named_parameters():
                torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)

    def test_centered_tangent_response_matches_finite_differences(self):
        for model in self.models():
            jacobian = access.prediction_jacobian(model, self.pairs)
            parameters = dict(model.named_parameters())
            direction = jacobian.T @ access.interaction(self.targets).flatten()
            direction /= direction.norm()
            expected = access.interaction((jacobian @ direction).reshape_as(self.targets))
            offset, delta = 0, {}
            for name, value in parameters.items():
                delta[name] = direction[offset:offset + value.numel()].reshape_as(value)
                offset += value.numel()
            step = 1e-5
            plus = functional_call(model, {name: value + step * delta[name] for name, value in parameters.items()},
                                   (self.pairs,))
            minus = functional_call(model, {name: value - step * delta[name] for name, value in parameters.items()},
                                    (self.pairs,))
            actual = access.interaction(((plus - minus) / (2 * step)).reshape_as(self.targets))
            torch.testing.assert_close(actual, expected, rtol=1e-7, atol=1e-10)

    def test_kernel_powers_match_actual_full_batch_gradient_steps(self):
        # Three observed rows, two parameters: retain the nontrivial nullspace.
        train = torch.tensor([[1., 0.], [0., 2.], [1., 1.]], dtype=torch.float64)
        heldout = torch.tensor([[2., -1.], [-1., 3.]], dtype=torch.float64)
        target = torch.tensor([1., -2., 4.], dtype=torch.float64)
        kernel, cross = train @ train.T / 3, heldout @ train.T / 3
        values, retained, path = access.linearized_path(kernel, cross, -target, range(9))
        self.assertEqual(int(retained.sum()), 2)
        parameters = torch.zeros(2, dtype=torch.float64)
        for residual, prediction in path:
            torch.testing.assert_close(residual, train @ parameters - target, rtol=1e-12, atol=1e-12)
            torch.testing.assert_close(prediction, heldout @ parameters, rtol=1e-12, atol=1e-12)
            parameters -= (train.T @ (train @ parameters - target)) / (3 * values[-1])


if __name__ == "__main__":
    unittest.main()
