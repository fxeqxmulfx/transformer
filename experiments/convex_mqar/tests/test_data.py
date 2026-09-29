"""Raw MQAR semantics, sampling, cache isolation, and split reproducibility."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from convex_mqar.certify import make_example
from convex_mqar.data import load_split, validate_batch

from .support import batch, config, recall_from_raw


class DataTests(unittest.TestCase):
    def test_generated_labels_follow_actual_previous_tokens(self):
        for length in (64, 128, 256, 512):
            tokens, positions, labels = batch(17, 5, length, 8192)
            validate_batch(tokens, positions, labels, 8192)
            for raw, queries, answers in zip(tokens.tolist(), positions.tolist(), labels.tolist()):
                for position, answer in zip(queries, answers):
                    self.assertEqual(recall_from_raw(raw, position), answer)

    def test_prefix_queries_and_fillers_have_the_documented_structure(self):
        tokens, positions, labels = batch(count=32)
        for raw, queries, answers in zip(tokens.tolist(), positions.tolist(), labels.tolist()):
            keys, values = raw[:8:2], raw[1:8:2]
            self.assertEqual(len(set(keys)), 4)
            self.assertEqual(len(set(queries)), 4)
            self.assertEqual(sorted(raw[p] for p in queries), sorted(keys))
            for j, key in enumerate(keys):
                self.assertLess(key, 32)
                self.assertGreaterEqual(values[j], 32)
                self.assertEqual(raw.count(key), 2)
            dictionary = dict(zip(keys, values))
            self.assertEqual(answers, [dictionary[raw[p]] for p in queries])
            for p in range(8, 16):
                if p not in queries:
                    self.assertGreaterEqual(raw[p], 32)

    def test_power_law_positions_are_sampled_without_replacement(self):
        class RecordingGenerator:
            def __init__(self):
                self.inner = np.random.default_rng(42)
                self.position_call = None

            def choice(self, candidates, **kwargs):
                if isinstance(candidates, np.ndarray):
                    self.position_call = candidates.copy(), kwargs.copy()
                return self.inner.choice(candidates, **kwargs)

            def integers(self, *args, **kwargs):
                return self.inner.integers(*args, **kwargs)

            def permutation(self, size):
                return self.inner.permutation(size)

        rng = RecordingGenerator()
        make_example(rng, 16, 4, 64, 0.7)
        available, arguments = rng.position_call
        np.testing.assert_array_equal(available, np.arange(8, 16))
        expected = np.array([p ** -0.7 for p in range(8, 16)])
        np.testing.assert_allclose(arguments["p"], expected / expected.sum())
        self.assertFalse(arguments["replace"])
        self.assertEqual(arguments["size"], 4)

    def test_values_are_resampled_between_sequences(self):
        tokens, _, _ = batch(29, 128, 8, 16)
        observed = {}
        for row in tokens.tolist():
            for key, value in zip(row[:4:2], row[1:4:2]):
                observed.setdefault(key, set()).add(value)
        self.assertTrue(all(len(values) > 1 for values in observed.values()))

    def test_cache_reload_and_fresh_generation_are_identical(self):
        cfg = config()
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            a, identity = load_split(first, cfg, 8, "train")
            cached, cached_identity = load_split(first, cfg, 8, "train")
            fresh, fresh_identity = load_split(second, cfg, 8, "train")
            self.assertEqual(identity, cached_identity)
            self.assertEqual(identity, fresh_identity)
            for left, right, new in zip(a, cached, fresh):
                torch.testing.assert_close(left, right)
                torch.testing.assert_close(left, new)

    def test_split_contents_are_independent_even_with_equal_sizes(self):
        cfg = replace(config(), train_examples=24, validation_examples=24, test_examples=24)
        with tempfile.TemporaryDirectory() as root:
            splits = [load_split(root, cfg, 8, split) for split in ("train", "validation", "test")]
        self.assertEqual(len({identity for _, identity in splits}), 3)
        for i in range(3):
            for j in range(i):
                self.assertFalse(torch.equal(splits[i][0][0], splits[j][0][0]))

    def test_cache_identity_changes_with_sampling_configuration(self):
        cfg = config()
        with tempfile.TemporaryDirectory() as root:
            _, identity = load_split(root, cfg, 8, "train")
            for changed in (replace(cfg, seed=1), replace(cfg, alpha=0.5),
                            replace(cfg, vocab=32), replace(cfg, train_examples=68)):
                _, new_identity = load_split(root, changed, 8, "train")
                self.assertNotEqual(new_identity, identity)

    def test_cache_rejects_mismatched_metadata(self):
        with tempfile.TemporaryDirectory() as root:
            cfg = config()
            load_split(root, cfg, 8, "train")
            metadata = next(Path(root).glob("*/metadata.json"))
            contents = json.loads(metadata.read_text())
            contents["alpha"] = 99
            metadata.write_text(json.dumps(contents))
            with self.assertRaisesRegex(RuntimeError, "cache"):
                load_split(root, cfg, 8, "train")

    def test_validator_rejects_wrong_labels(self):
        tokens, positions, labels = batch()
        labels[0, 0] = (labels[0, 0] - 32 + 1) % 32 + 32
        with self.assertRaisesRegex(AssertionError, "Labels"):
            validate_batch(tokens, positions, labels, 64)

    def test_validator_rejects_noncausal_and_missing_key_queries(self):
        tokens, positions, labels = batch()
        positions[0, 0], labels[0, 0] = 0, tokens[0, 1]
        with self.assertRaisesRegex(AssertionError, "precedes"):
            validate_batch(tokens, positions, labels, 64)
        tokens, positions, labels = batch()
        tokens[0, positions[0, 0]] = 32
        with self.assertRaisesRegex(AssertionError, "prior key"):
            validate_batch(tokens, positions, labels, 64)

    def test_validator_rejects_duplicate_keys(self):
        tokens, positions, labels = batch()
        tokens[0, 2] = tokens[0, 0]
        with self.assertRaisesRegex(AssertionError, "prior key"):
            validate_batch(tokens, positions, labels, 64)
