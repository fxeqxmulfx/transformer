"""Known group characters distinguish frequency concentration from position separation."""

import math
import unittest

import torch

from lab.infrastructure.rotary_geometry import embedding_geometry, multiplicative_ordering


class RotaryGeometryTests(unittest.TestCase):
    def character(self, separated):
        generator, residues = multiplicative_ordering(97)
        self.assertEqual(generator, 5)
        self.assertEqual(set(residues), set(range(1, 97)))
        for i in range(96):
            self.assertEqual(residues[(i + 1) % 96], residues[i] * generator % 97)
        order = torch.tensor(residues)
        phase = torch.arange(96, dtype=torch.float64) * (2 * math.pi * 7 / 96)
        embedding = torch.zeros(97, 6, dtype=torch.float64)
        embedding[order, 0] = phase.cos()
        embedding[order, 2 if separated else 1] = phase.sin()
        return embedding, order

    def test_a_faithful_character_in_separate_operand_channels(self):
        embedding, order = self.character(separated=True)
        # A common bias and the zero-class row are not character information.
        embedding[order, 4] = 3
        embedding[0] = 1000
        geometry = embedding_geometry(embedding, order)
        self.assertAlmostEqual(geometry["effective_rank"], 2, places=12)
        self.assertAlmostEqual(geometry["J_overlap"], 0, places=12)
        self.assertAlmostEqual(geometry["mean_energy_fraction"], 0.9, places=12)
        self.assertAlmostEqual(sum(geometry["character_pair_shares"]), 1, places=12)
        self.assertAlmostEqual(geometry["largest_character_pair_share"], 1, places=12)
        self.assertEqual(geometry["dominant_character_frequency"], 7)
        self.assertEqual(geometry["dominant_character_order"], 96)

    def test_a_rotation_invariant_character_plane_has_full_J_overlap(self):
        embedding, order = self.character(separated=False)
        geometry = embedding_geometry(embedding, order)
        self.assertAlmostEqual(geometry["effective_rank"], 2, places=12)
        self.assertAlmostEqual(geometry["J_overlap"], 1, places=12)
        self.assertAlmostEqual(geometry["largest_character_pair_share"], 1, places=12)

    def test_constant_and_trivial_embeddings_have_no_centered_character(self):
        for prime in (2, 5):
            with self.subTest(prime=prime):
                _, order = multiplicative_ordering(prime)
                geometry = embedding_geometry(torch.ones(prime, 4), torch.tensor(order))
                self.assertEqual(geometry["mean_energy_fraction"], 1)
                self.assertIsNone(geometry["effective_rank"])
                self.assertIsNone(geometry["J_overlap"])
                self.assertIsNone(geometry["dominant_character_frequency"])
                self.assertEqual(geometry["character_pair_shares"], [])
