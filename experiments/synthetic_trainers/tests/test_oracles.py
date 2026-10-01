"""Compare causal labels with independent stack and regular-language semantics."""

from dataclasses import replace
import itertools
import re
import unittest

from experiments.synthetic_trainers import vocabulary as v
from experiments.synthetic_trainers.oracles import lookup_targets, oracle_targets, prefix_targets, validate_example
from experiments.synthetic_trainers.records import example_from_targets
from experiments.synthetic_trainers.specs import TaskSpec


def stack_status(word):
    stack = []
    for token in word:
        if token == v.OPEN:
            stack.append(token)
        elif token == v.CLOSE:
            if not stack:
                return v.INVALID
            stack.pop()
    return v.INCOMPLETE if stack else v.BALANCED


def block_membership(word, blocks):
    active = "".join("a" if token == v.A else "b" for token in word if token != v.NEUTRAL)
    pattern = "".join("a+" if index % 2 == 0 else "b+" for index in range(blocks))
    return v.ACCEPT if re.fullmatch(pattern, active) else v.REJECT


class OracleTests(unittest.TestCase):
    def test_dyck_against_stack_for_all_short_words_and_prefixes(self):
        for size in range(6):
            for word in itertools.product((v.OPEN, v.CLOSE, v.NEUTRAL), repeat=size):
                expected = [v.IGNORE] + [stack_status(word[:end]) for end in range(1, size + 1)]
                self.assertEqual(prefix_targets((v.BOS, *word), "dyck"), expected)

    def test_blocks_against_regular_language_for_all_short_words(self):
        for blocks in range(1, 5):
            for word in itertools.product((v.A, v.B, v.NEUTRAL), repeat=5):
                expected = [v.IGNORE] + [block_membership(word[:end], blocks) for end in range(1, 6)]
                self.assertEqual(prefix_targets((v.BOS, *word), "blocks", blocks), expected)

    def test_invalid_dyck_prefix_never_recovers_after_balance_returns(self):
        self.assertEqual(prefix_targets((v.BOS, v.CLOSE, v.OPEN, v.NEUTRAL), "dyck"),
                         [v.IGNORE, v.INVALID, v.INVALID, v.INVALID])

    def test_neutrals_and_future_suffix_do_not_change_earlier_labels(self):
        for task, first, second in (("dyck", v.OPEN, v.CLOSE), ("blocks", v.A, v.B)):
            prefix = (v.BOS, first, v.NEUTRAL, first, second)
            expected = prefix_targets(prefix, task)
            for suffix in itertools.product((first, second, v.NEUTRAL), repeat=4):
                self.assertEqual(prefix_targets((*prefix, *suffix), task)[:len(prefix)], expected)

    def test_three_hop_lookup_from_shuffled_records(self):
        a, b, c, d = range(v.IDENTITY_BASE, v.IDENTITY_BASE + 4)
        tokens = (v.BOS, v.KEY, c, v.VALUE, d, v.KEY, a, v.VALUE, b,
                  v.KEY, b, v.VALUE, c, v.END_TABLE, v.FILL, v.QUERY, a)
        targets = lookup_targets(tokens, 3)
        self.assertEqual(targets[:-1], [v.IGNORE] * (len(tokens) - 1))
        self.assertEqual(targets[-1], d)

    def test_lookup_rejects_future_links_missing_links_and_cycles(self):
        a, b = v.IDENTITY_BASE, v.IDENTITY_BASE + 1
        cases = (
            (v.BOS, v.QUERY, a, v.KEY, a, v.VALUE, b, v.END_TABLE),
            (v.BOS, v.KEY, a, v.VALUE, b, v.END_TABLE, v.QUERY, a),
            (v.BOS, v.KEY, a, v.VALUE, a, v.END_TABLE, v.QUERY, a),
        )
        for raw in cases:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                lookup_targets(raw, 2)

    def test_duplicate_keys_and_queries_are_rejected(self):
        a, b = v.IDENTITY_BASE, v.IDENTITY_BASE + 1
        for raw in (
            (v.BOS, v.KEY, a, v.VALUE, b, v.KEY, a, v.VALUE, b, v.END_TABLE),
            (v.BOS, v.KEY, a, v.VALUE, b, v.END_TABLE, v.QUERY, a, v.QUERY, a),
        ):
            with self.assertRaises(ValueError):
                lookup_targets(raw, 1)

    def test_validator_rejects_corrupted_labels_and_masks(self):
        spec = TaskSpec(task="blocks", length=7, blocks=2, neutral_fraction=0)
        raw = (v.BOS, v.A, v.A, v.A, v.B, v.B, v.B)
        example = example_from_targets("blocks", raw, oracle_targets(raw, spec))
        validate_example(example, spec)
        for corrupted in (
            replace(example, targets=(*example.targets[:-1], v.REJECT)),
            replace(example, changes=(*example.changes[:-1], True)),
            replace(example, task="dyck"),
        ):
            with self.assertRaises(ValueError):
                validate_example(corrupted, spec)

    def test_invalid_prefix_alphabet_is_rejected(self):
        for task in ("dyck", "blocks"):
            with self.assertRaises(ValueError):
                prefix_targets((v.BOS, v.ACCEPT), task)
