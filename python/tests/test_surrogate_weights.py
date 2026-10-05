"""EXPERIMENT_PLAN.md, step 3: separate routing from score sensitivity.

The independent sparsemax reference uses regular tensor operations for
Algorithm 1 of arXiv:1602.02068v2, section 2.2. torch.func differentiates
its sort and threshold; it does not call the production custom backward.
At an exact threshold it selects the strictly-positive-support Jacobian
of section 2.4. A mixed surrogate deliberately fails a finite-difference
test of its own forward: its specified backward belongs to the other map.
"""

import unittest

import torch
from torch.nn import functional as F

from lab.dsl import (PerHeadQKV, QKNorm, RMSNorm, ScaledDot, Softmax, Sparsemax,
                     SurrogateWeights, describe, substitute, swap)
from lab.infrastructure.nn import build_model
from lab.infrastructure.nn.attention import weights_function
from lab.infrastructure.nn.sparsemax import causal_sparsemax

from examples import gptmini


def reference_softmax(scores):
    length = scores.shape[-1]
    future = torch.ones(length, length, dtype=torch.bool, device=scores.device).triu(1)
    return torch.softmax(scores.masked_fill(future, -torch.inf), dim=-1)


def reference_sparsemax(scores):
    """Paper Algorithm 1, without the production max shift or custom autograd Function."""
    length = scores.shape[-1]
    future = torch.ones(length, length, dtype=torch.bool, device=scores.device).triu(1)
    stable = scores if scores.dtype == torch.float64 else scores.float()
    visible = stable.masked_fill(future, -torch.inf)
    ordered = visible.sort(descending=True, dim=-1).values
    ranks = torch.arange(1, length + 1, device=scores.device, dtype=stable.dtype)
    sums = ordered.cumsum(-1)
    size = (1 + ranks * ordered > sums).sum(-1, keepdim=True)
    threshold = (sums.gather(-1, size - 1) - 1) / size
    difference = visible - threshold
    return torch.where(difference > 0, difference, torch.zeros_like(difference))


MIXED = ((Sparsemax(), Softmax(), reference_softmax),
         (Softmax(), Sparsemax(), reference_sparsemax))


def model_pair(normalizer, *, scores=QKNorm(), projections=None):
    spec = swap(gptmini(width=16, depth=2, heads=2), 'context', 7)
    spec = swap(spec, 'block.attention.scores', scores)
    if projections is not None:
        spec = swap(spec, 'block.attention.projections', projections)
    plain = substitute(spec, Softmax, normalizer)
    diagonal = substitute(spec, Softmax, SurrogateWeights(normalizer, normalizer))
    return [build_model(model, vocab=8, seed=9) for model in (plain, diagonal)]


def logits_and_gradients(model, tokens, targets):
    logits = model(tokens)
    F.cross_entropy(logits.flatten(0, 1), targets.flatten()).backward()
    original = getattr(model, '_orig_mod', model)
    return logits, [(name, parameter.grad) for name, parameter in original.named_parameters()]


class SurrogateWeightsTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch._dynamo.reset()
        self.generator = torch.Generator().manual_seed(173)
        self.tokens = torch.randint(8, (2, 7), generator=self.generator)
        self.targets = torch.randint(8, (2, 7), generator=self.generator)

    def compare_models(self, pair, *, exact):
        (expected, expected_gradients), (actual, actual_gradients) = [
            logits_and_gradients(model, self.tokens, self.targets) for model in pair]
        if exact:
            self.assertTrue(torch.equal(actual, expected))
        else:
            torch.testing.assert_close(actual, expected)
        for (name, expected), (actual_name, actual) in zip(expected_gradients, actual_gradients, strict=True):
            with self.subTest(parameter=name):
                self.assertEqual(name, actual_name)
                self.assertIsNotNone(actual)
                self.assertIsNotNone(expected)
                if exact:
                    self.assertTrue(torch.equal(actual, expected))
                else:
                    torch.testing.assert_close(actual, expected)

    def test_block_describes_both_maps_and_rejects_unsupported_compositions(self):
        self.assertEqual(describe(SurrogateWeights(Sparsemax(), Softmax())),
                         {'type': 'SurrogateWeights', 'forward': {'type': 'Sparsemax'},
                          'backward': {'type': 'Softmax', 'fused': False}})
        for forward, backward in ((RMSNorm(), Softmax()), (Softmax(), RMSNorm())):
            with self.subTest(forward=forward, backward=backward), self.assertRaises(TypeError):
                SurrogateWeights(forward, backward)
        nested = SurrogateWeights(Softmax(), Sparsemax())
        with self.assertRaises(ValueError):
            SurrogateWeights(nested, Softmax())
        for forward, backward in ((Softmax(fused=True), Sparsemax()),
                                  (Sparsemax(), Softmax(fused=True))):
            with self.subTest(forward=forward, backward=backward), self.assertRaises(ValueError):
                SurrogateWeights(forward, backward)
        self.assertEqual(SurrogateWeights(Softmax(fused=True), Softmax(fused=True)).forward,
                         Softmax(fused=True))

    def test_diagonal_logits_and_every_gradient_are_bit_identical(self):
        for normalizer in (Softmax(), Sparsemax(), Softmax(fused=True)):
            for scores in (QKNorm(), ScaledDot()):
                for projections in (None, PerHeadQKV()):
                    with self.subTest(normalizer=normalizer, scores=scores, projections=projections):
                        self.compare_models(model_pair(normalizer, scores=scores, projections=projections), exact=True)

    def test_diagonal_compilation_preserves_bit_identical_logits_and_gradients(self):
        for normalizer in (Softmax(), Sparsemax()):
            with self.subTest(normalizer=normalizer):
                pair = [torch.compile(model, fullgraph=True, dynamic=False) for model in model_pair(normalizer)]
                self.compare_models(pair, exact=True)

    def test_mixed_forward_is_exact_and_backward_matches_independent_vjp_including_inactive_entries(self):
        for dtype in (torch.float64, torch.float32, torch.float16, torch.bfloat16):
            source = torch.randn(2, 3, 7, 7, generator=self.generator, dtype=dtype)
            source[..., -1, :] = torch.tensor([2., 1., 0., -1., -2., -3., -4.], dtype=dtype)
            upstream = torch.randn(source.shape, generator=self.generator, dtype=dtype)
            future = torch.ones(7, 7, dtype=torch.bool).triu(1).expand_as(source)
            sparse_support = causal_sparsemax(source) > 0
            for forward, backward, reference in MIXED:
                with self.subTest(dtype=dtype, forward=forward, backward=backward):
                    scores = source.clone().requires_grad_()
                    actual = weights_function(SurrogateWeights(forward, backward))(scores)
                    expected_weights = weights_function(forward)(source)
                    self.assertEqual(actual.dtype, expected_weights.dtype)
                    self.assertTrue(torch.equal(actual, expected_weights))
                    actual.backward(upstream.to(actual.dtype))
                    reference_weights, vjp = torch.func.vjp(reference, source)
                    (expected_gradient,) = vjp(upstream.to(reference_weights.dtype))
                    torch.testing.assert_close(scores.grad, expected_gradient)
                    self.assertEqual(torch.count_nonzero(scores.grad[future]).item(), 0)
                    if isinstance(backward, Sparsemax):
                        self.assertEqual(torch.count_nonzero(scores.grad[~sparse_support]).item(), 0)
                        self.assertEqual(torch.count_nonzero(scores.grad[..., -1, :]).item(), 0)
                    else:
                        self.assertGreater(torch.count_nonzero(scores.grad[~sparse_support & ~future]).item(), 0)

    def test_mixed_model_logits_are_bit_identical_to_the_forward_normalizer(self):
        spec = swap(gptmini(width=16, depth=2, heads=2), 'context', 7)
        for forward, backward, _ in MIXED:
            plain, mixed = [build_model(substitute(spec, Softmax, normalizer), vocab=8, seed=9)
                            for normalizer in (forward, SurrogateWeights(forward, backward))]
            with self.subTest(forward=forward, backward=backward), torch.no_grad():
                self.assertTrue(torch.equal(mixed(self.tokens), plain(self.tokens)))

    def test_mixed_full_graph_compilation_preserves_logits_and_every_parameter_gradient(self):
        spec = swap(gptmini(width=16, depth=2, heads=2), 'context', 7)
        for forward, backward, _ in MIXED:
            mixed = substitute(spec, Softmax, SurrogateWeights(forward, backward))
            eager, compiled = [build_model(mixed, vocab=8, seed=9) for _ in range(2)]
            with self.subTest(forward=forward, backward=backward):
                self.compare_models([eager, torch.compile(compiled, fullgraph=True, dynamic=False)], exact=False)


if __name__ == '__main__':
    unittest.main()
