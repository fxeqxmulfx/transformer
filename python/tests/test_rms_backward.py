"""Independent finite-epsilon derivatives, recovery and compiled-state checks."""

import math
import unittest

import torch
from torch.nn import functional as F

from lab.dsl import AccurateBranchGates, BranchGates
from lab.infrastructure.nn.gradient_paths import build
from lab.infrastructure.nn.rms_backward import rms_norm

EPS = 2 ** -23


class RMSBackwardTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_forward_is_identical_to_native_at_all_scales(self):
        generator = torch.Generator().manual_seed(127)
        for shape in ((2, 32), (4, 8, 32)):
            for scale in (.001, 1., 1000.):
                x = torch.randn(shape, generator=generator) * scale
                self.assertTrue(torch.equal(rms_norm(x, 32, EPS), F.rms_norm(x, (32,), eps=EPS)))

    def test_scalar_chain_matches_independent_real_derivative(self):
        for scale in (.001, 1., 1000.):
            logit = torch.tensor(0., requires_grad=True)
            x = (2 * logit.sigmoid() * scale) * torch.ones(1, 32)
            output = rms_norm(x, 32, EPS)
            derivative, = torch.autograd.grad((output - .25).square().mean(), logit)
            amplitude = scale / math.sqrt(scale * scale + EPS)
            expected = (amplitude - .25) * scale * EPS / (scale * scale + EPS) ** 1.5
            self.assertGreater(derivative.item(), 0.)
            self.assertAlmostEqual(derivative.item() / expected, 1., delta=2e-6)
        # This rejects the native backward, not just a changed forward function.
        logit = torch.tensor(0., requires_grad=True)
        output = F.rms_norm(2 * logit.sigmoid() * torch.ones(1, 32), (32,), eps=EPS)
        derivative, = torch.autograd.grad((output - .25).square().mean(), logit)
        self.assertEqual(derivative.item(), 0.)

    def test_vector_vjp_matches_independent_fp64_autograd(self):
        generator = torch.Generator().manual_seed(91)
        for scale in (.001, 1., 1000.):
            x = (torch.randn(3, 2, 32, generator=generator) * scale).requires_grad_()
            upstream = torch.randn(x.shape, generator=generator)
            gradient, = torch.autograd.grad(rms_norm(x, 32, EPS), x, upstream)
            reference = x.detach().double().requires_grad_()
            expected, = torch.autograd.grad(F.rms_norm(reference, (32,), eps=EPS), reference, upstream.double())
            torch.testing.assert_close(gradient.double(), expected, rtol=2e-6, atol=1e-10)

    def test_zero_input_and_higher_derivatives(self):
        x = torch.zeros(2, 4, dtype=torch.float64, requires_grad=True)
        upstream = torch.arange(8, dtype=torch.float64).reshape(2, 4)
        gradient, = torch.autograd.grad(rms_norm(x, 4, EPS), x, upstream)
        torch.testing.assert_close(gradient, upstream / math.sqrt(EPS))
        sample = torch.tensor([[.2, -.4, .7, .9]], dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(lambda z: rms_norm(z, 4, EPS), sample))
        self.assertTrue(torch.autograd.gradgradcheck(lambda z: rms_norm(z, 4, EPS), sample))

    def test_compiled_branch_restores_radial_gradient(self):
        native = build(BranchGates(placement="before", scale=1.))
        accurate = build(AccurateBranchGates(placement="before", scale=1.))
        inputs = torch.ones(64, 2, 32)
        outputs = []
        gradients = []
        for model in (native, accurate):
            compiled = torch.compile(model, mode="max-autotune", fullgraph=True, dynamic=False)
            output = compiled(inputs)
            (output - 1.25).square().mean().backward()
            outputs.append(output.detach())
            gradients.append(model.logits.grad.item())
        self.assertTrue(torch.equal(*outputs))
        self.assertEqual(gradients[0], 0.)
        expected = .75 * EPS / (1 + EPS) ** 1.5
        self.assertGreater(gradients[1], 0.)
        self.assertAlmostEqual(gradients[1] / expected, 1., delta=2e-6)


if __name__ == "__main__":
    unittest.main()
