"""Strict public dataset parsing and independent complex least-squares checks."""

import struct
import unittest

import numpy as np
import torch

from experiments.synthetic_trainers.paper_reproduction.fashion_data import images_from_idx, labels_from_idx
from experiments.synthetic_trainers.paper_reproduction.random_features import features, finite_gradient_flow, fit_head


class FashionDataTests(unittest.TestCase):
    def test_idx_headers_payload_sizes_and_label_domain(self):
        images = struct.pack(">IIII", 2051, 2, 28, 28) + bytes(range(256)) * 6 + bytes(range(32))
        self.assertEqual(images_from_idx(images).shape, (2, 784))
        self.assertEqual(labels_from_idx(struct.pack(">II", 2049, 2) + bytes((0, 9))).tolist(), [0, 9])
        for damaged in (images[:-1], b"", struct.pack(">IIII", 2051, 1, 27, 28) + bytes(27 * 28)):
            with self.assertRaises(ValueError):
                images_from_idx(damaged)
        with self.assertRaises(ValueError):
            labels_from_idx(struct.pack(">II", 2049, 1) + bytes((10,)))


class RandomFeatureTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_complex_activation_and_variance_scaling(self):
        images = torch.eye(2, dtype=torch.float64)
        gaussian = torch.diag(torch.tensor([np.pi / 2, np.pi], dtype=torch.float64)) * 2 ** .5
        measured = features(images, gaussian, 2)
        expected = torch.tensor([[-1j, 1], [1, -1]], dtype=torch.complex128)
        torch.testing.assert_close(measured, expected, rtol=0, atol=1e-14)

    def test_underdetermined_head_interpolates_with_minimum_norm(self):
        design = torch.tensor([[1, 1j, 0], [0, 1, 1]], dtype=torch.complex128)
        labels = torch.tensor([0, 1])
        head, diagnostics = fit_head(design, labels, classes=2)
        target = torch.eye(2, dtype=torch.complex128)
        torch.testing.assert_close(design @ head, target, rtol=0, atol=1e-14)
        torch.testing.assert_close(head, torch.linalg.pinv(design) @ target, rtol=1e-13, atol=1e-13)
        null = torch.tensor([-1j, 1, -1], dtype=torch.complex128)
        torch.testing.assert_close(head.mH @ null, torch.zeros(2, dtype=torch.complex128), rtol=0, atol=1e-14)
        self.assertLess(diagnostics["normal_equation_residual"], 1e-13)
        torch.testing.assert_close(finite_gradient_flow(design, labels, 1000, classes=2), head, rtol=1e-13, atol=1e-13)
        self.assertEqual(finite_gradient_flow(design, labels, 0, classes=2).abs().max().item(), 0)

    def test_overdetermined_solution_matches_SVD_least_squares(self):
        generator = torch.Generator().manual_seed(1)
        design = torch.randn(6, 2, dtype=torch.complex128, generator=generator)
        labels = torch.tensor([0, 1, 0, 1, 0, 1])
        target = torch.nn.functional.one_hot(labels, 2).to(torch.complex128)
        head, diagnostics = fit_head(design, labels, classes=2)
        expected = torch.linalg.lstsq(design, target, driver="gelsd").solution
        torch.testing.assert_close(head, expected, rtol=1e-12, atol=1e-12)
        self.assertGreater(diagnostics["interpolation_MSE"], .01)
        self.assertLess(diagnostics["normal_equation_residual"], 1e-13)


