"""Step 0 of EXPERIMENT_PLAN.md: compiled sparsemax and diagnostic noninterference.

The simplex projection and its support Jacobian follow arXiv:1602.02068v2,
sections 2.2 and 2.4. The compiler must preserve both, including causal
zeros; sampling compiled updates must preserve the complete training state.
"""

from pathlib import Path
import tempfile
import unittest

import torch
from torch.nn import functional as F

from lab.domain.model import Softmax, Sparsemax
from lab.domain.spec import substitute, swap
from lab.domain.training import Compiled, Diagnostics
from lab.infrastructure.engine.compiled import CompiledStepper
from lab.infrastructure.nn import build_model
from lab.infrastructure.nn.sparsemax import causal_sparsemax
from lab.infrastructure.store import RunDirectory

from examples import gptmini, modular
from test_engine import untimed
from test_graphs import summary, train


class SparsemaxCompiledTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch._dynamo.reset()

    def test_compiled_projection_and_backward_preserve_the_simplex_and_causal_zeros(self):
        compiled = torch.compile(causal_sparsemax, dynamic=False, fullgraph=True)
        generator = torch.Generator().manual_seed(31)
        for length in (1, 17, 64):
            with self.subTest(length=length):
                source = torch.randn(2, 4, length, length, generator=generator)
                upstream = torch.randn(source.shape, generator=generator)
                eager_scores, compiled_scores = (source.clone().requires_grad_() for _ in range(2))
                eager, actual = causal_sparsemax(eager_scores), compiled(compiled_scores)
                eager.backward(upstream)
                actual.backward(upstream)
                torch.testing.assert_close(actual, eager)
                torch.testing.assert_close(compiled_scores.grad, eager_scores.grad)
                torch.testing.assert_close(actual.sum(-1), torch.ones_like(actual.sum(-1)))
                self.assertTrue(bool((actual >= 0).all()))
                self.assertEqual(torch.count_nonzero(actual.triu(1)).item(), 0)
                self.assertEqual(torch.count_nonzero(compiled_scores.grad.triu(1)).item(), 0)

    def test_compiled_sparsemax_logits_and_every_parameter_gradient_match_eager(self):
        spec = substitute(gptmini(64, 2, 4), Softmax, Sparsemax())
        spec = swap(spec, "context", 64)
        eager, compiled_model = (build_model(spec, vocab=32, seed=0) for _ in range(2))
        compiled = torch.compile(compiled_model, dynamic=False, fullgraph=True)
        generator = torch.Generator().manual_seed(53)
        tokens = torch.randint(32, (2, 64), generator=generator)
        targets = torch.randint(32, (2, 64), generator=generator)
        expected, actual = eager(tokens), compiled(tokens)
        F.cross_entropy(expected.flatten(0, 1), targets.flatten()).backward()
        F.cross_entropy(actual.flatten(0, 1), targets.flatten()).backward()
        torch.testing.assert_close(actual, expected)
        for (name, parameter), (compiled_name, compiled_parameter) in zip(
                eager.named_parameters(), compiled_model.named_parameters(), strict=True):
            with self.subTest(parameter=name):
                self.assertEqual(name, compiled_name)
                torch.testing.assert_close(compiled_parameter.grad, parameter.grad)

    def test_diagnostics_preserve_compiled_losses_and_training_state_under_both_weights(self):
        base = modular(gptmini(32, 2, 4), prime=11, updates=20, batch=8, every=10,
                       execution=Compiled())
        for weights in (Softmax(), Sparsemax()):
            run = substitute(base, Softmax, weights)
            with self.subTest(weights=type(weights).__name__), tempfile.TemporaryDirectory() as root:
                plain, sampled = Path(root) / "plain", Path(root) / "sampled"
                train(run, plain, CompiledStepper)
                train(swap(run, "diagnostics", Diagnostics(every=10)), sampled, CompiledStepper)
                plain, sampled = RunDirectory(plain), RunDirectory(sampled)
                self.assertEqual(untimed(plain.records("history")), untimed(sampled.records("history")))
                self.assertEqual(summary(plain.checkpoint("cpu")), summary(sampled.checkpoint("cpu")))
                self.assertEqual(summary(plain.result()), summary(sampled.result()))
                measurements = sampled.records("diagnostics")
                self.assertEqual([row["step"] for row in measurements], [10, 20])
                self.assertTrue(all(row["temperatures"] for row in measurements))
                self.assertTrue(all("answer_loss" in row and "EOS_loss" in row for row in measurements))


if __name__ == "__main__":
    unittest.main()
