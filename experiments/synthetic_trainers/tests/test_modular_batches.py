"""The tail control keeps the shuffled stream and explicitly changes exposure."""

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import torch

from experiments.synthetic_trainers.paper_reproduction.batches import next_batch
from experiments.synthetic_trainers.paper_reproduction.grokking import RunConfig, train
from experiments.synthetic_trainers.tests.test_modular_diagnostics import assert_checkpoint_equal


def stream(policy, n=21, batch_size=8, updates=15):
    generator = torch.Generator().manual_seed(123)
    permutation = torch.randperm(n, generator=generator)
    cursor, batches, metadata = 0, [], []
    for _ in range(updates):
        rows, permutation, cursor, info = next_batch(permutation, cursor, generator, batch_size, policy)
        batches.append(rows)
        metadata.append(info)
    return batches, metadata


class ModularBatchTests(unittest.TestCase):
    def test_short_tail_matches_independently_shuffled_legacy_batches(self):
        actual, metadata = stream("short_final")
        generator = torch.Generator().manual_seed(123)
        expected = []
        for _ in range(5):
            permutation = torch.randperm(21, generator=generator)
            expected.extend(permutation[start:start + 8] for start in range(0, 21, 8))
        self.assertEqual([len(row) for row in actual], [8, 8, 5] * 5)
        for left, right in zip(actual, expected):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        self.assertEqual([info["epoch_tail"] for info in metadata], [False, False, True] * 5)

    def test_full_batches_use_the_same_stream_including_multi_epoch_wraps(self):
        for batch_size in (8, 50):
            with self.subTest(batch_size=batch_size):
                full, metadata = stream("wrap_epoch", batch_size=batch_size)
                short, _ = stream("short_final", batch_size=batch_size, updates=45)
                expected = torch.cat(short)[:len(torch.cat(full))]
                torch.testing.assert_close(torch.cat(full), expected, rtol=0, atol=0)
                self.assertTrue(all(len(row) == batch_size for row in full))
                self.assertTrue(any(info["epoch_wraps_in_batch"] for info in metadata))

    def test_wrap_control_resumes_exactly_and_reports_actual_exposure(self):
        config = RunConfig(model="gptmini", optimizer="amsgradw", prime=7,
                           train_fraction=.5, width=8, heads=1, layers=1, steps=6,
                           eval_every=1, batch_size=8, batch_policy="wrap_epoch", device="cpu")
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            full = train(config, root / "full")
            train(replace(config, steps=3), root / "resumed")
            train(config, root / "resumed", resume=True)
            assert_checkpoint_equal(self, torch.load(root / "full/checkpoint.pt", weights_only=True),
                                    torch.load(root / "resumed/checkpoint.pt", weights_only=True))
            self.assertAlmostEqual(full["final"]["epochs_seen"], 48 / 21)
            self.assertTrue(any("more_examples" in item for item in full["plan"]["deviations"]))
            with self.assertRaises(ValueError):
                train(replace(config, batch_policy="short_final"), root / "resumed", resume=True)
