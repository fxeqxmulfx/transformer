"""Post-run audit of the measured Muon hybrid and parameter routing.

These tests preserve the frozen benchmark implementations. The 0.05 auxiliary
LR ratio is an experimental choice, not the shared-LR recipe of Muon §2.2.
"""

import math
import unittest

import torch

from experiments.gpt_mini import Config
from experiments.optimizer_benchmark.attention import make_model
from experiments.optimizer_benchmark.matrix import MuonOptimizer


class MuonParameterGroupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_actual_model_routes_hidden_matrices_and_unique_auxiliary_weights(self):
        config = Config(vocab_size=65, n_layers=2, n_heads=4,
                        d_model=128, d_ff=512, max_seq_len=64)
        model = make_model(config, "softmax", 0)
        optimizer = MuonOptimizer(model.named_parameters(), 0.03)
        for parameter in model.parameters():
            parameter.grad = torch.full_like(parameter, 0.001)
        optimizer.step()
        hidden, auxiliary = [], []
        for name, parameter in model.named_parameters():
            state = optimizer.state[parameter]
            if parameter.ndim == 2 and name != "embed.weight":
                self.assertIn("momentum", state)
                self.assertNotIn("v", state)
                hidden.append(parameter.numel())
            else:
                self.assertIn("m", state)
                self.assertIn("v", state)
                self.assertNotIn("momentum", state)
                auxiliary.append(parameter.numel())
        self.assertEqual((len(hidden), sum(hidden)), (8, 393216))
        self.assertEqual((len(auxiliary), sum(auxiliary)), (3, 8328))
        self.assertIs(model.embed.weight, model.unembed.weight)
        self.assertEqual(sum(p is model.embed.weight for p in optimizer.param_groups[0]["params"]), 1)

    def test_auxiliary_updates_match_independent_torch_adamw_with_zero_decay(self):
        embedding = torch.nn.Parameter(torch.zeros(2, 2, dtype=torch.float64))
        temperature = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))
        reference = [torch.nn.Parameter(p.detach().clone()) for p in (embedding, temperature)]
        optimizer = MuonOptimizer([("embed.weight", embedding), ("blocks.0.attn.log_alpha", temperature)], 0.03)
        adam = torch.optim.AdamW(reference, lr=0.0015, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
        for step in range(2):
            for parameter, copy in zip((embedding, temperature), reference):
                gradient = torch.arange(1, parameter.numel() + 1, dtype=parameter.dtype).reshape_as(parameter)
                gradient = gradient * (0.1 if step == 0 else -0.07)
                parameter.grad, copy.grad = gradient.clone(), gradient.clone()
            optimizer.step()
            adam.step()
            for actual, expected in zip((embedding, temperature), reference):
                torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)

    def test_matrix_updates_match_independent_scalar_nesterov_and_polynomial(self):
        parameter = torch.nn.Parameter(torch.eye(2, dtype=torch.float64))
        optimizer = MuonOptimizer([("blocks.0.ffn.w_in.weight", parameter)], 0.03)
        momentum = [0.0, 0.0]
        expected = [1.0, 1.0]
        for gradient in ((0.3, 0.1), (-0.2, 0.4)):
            parameter.grad = torch.diag(torch.tensor(gradient, dtype=parameter.dtype))
            momentum = [0.95 * m + g for m, g in zip(momentum, gradient)]
            candidate = [0.95 * m + g for m, g in zip(momentum, gradient)]
            norm = math.sqrt(sum(x * x for x in candidate))
            singular = [x / norm for x in candidate]
            for _ in range(5):
                singular = [3.4445 * x - 4.775 * x**3 + 2.0315 * x**5 for x in singular]
            expected = [x - 0.03 * 0.2 * math.sqrt(2) * d for x, d in zip(expected, singular)]
            optimizer.step()
            torch.testing.assert_close(parameter, torch.diag(torch.tensor(expected, dtype=parameter.dtype)),
                                       rtol=1e-12, atol=1e-12)
            torch.testing.assert_close(optimizer.state[parameter]["momentum"],
                                       torch.diag(torch.tensor(momentum, dtype=parameter.dtype)))

    def test_actual_hybrid_guard_rejects_joint_candidate_and_preserves_state(self):
        config = Config(vocab_size=8, n_layers=1, n_heads=2, d_model=16, d_ff=32, max_seq_len=8)
        model = make_model(config, "sparsemax", 0)
        optimizer = MuonOptimizer(model.named_parameters(), 0.1, guarded=True)
        before = [p.detach().clone() for p in model.parameters()]
        for parameter in model.parameters():
            parameter.grad = torch.full_like(parameter, 1e-5)
        optimizer.step()
        self.assertEqual(optimizer.guard_accepted, 0)
        for parameter, initial in zip(model.parameters(), before):
            torch.testing.assert_close(parameter, initial - 0.1 * parameter.grad, rtol=0, atol=1e-8)
            self.assertTrue(bool(optimizer.state[parameter]))


if __name__ == "__main__":
    unittest.main()
