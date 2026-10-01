"""Generator contracts, matched examples, difficulty controls, and frozen splits."""

from collections import Counter
from dataclasses import replace
import json
import random
import tempfile
from pathlib import Path
import unittest

from experiments.synthetic_trainers import vocabulary as v
from experiments.synthetic_trainers.data import build_split
from experiments.synthetic_trainers.lookup import lookup_pair
from experiments.synthetic_trainers.oracles import validate_example
from experiments.synthetic_trainers.prefixes import prefix_pair
from experiments.synthetic_trainers.specs import TASKS, TaskSpec


class DataTests(unittest.TestCase):
    def test_mqar_labels_from_independent_previous_occurrence_oracle(self):
        spec = TaskSpec(task="mqar", length=64, min_length=32, pairs=8, queries=8, query_gap=2)
        split = build_split(spec, "train", 9, 64)
        for example in split.examples:
            queries = [i for i, target in enumerate(example.targets) if target != v.IGNORE]
            self.assertEqual(len(queries), 8)
            for position in queries:
                previous = [i for i in range(position - 1) if example.tokens[i] == example.tokens[position]]
                self.assertEqual(example.targets[position], example.tokens[previous[-1] + 1])
            self.assertGreaterEqual(min(queries), 1 + 2 * spec.pairs + spec.query_gap)

    def test_lookup_generated_answers_follow_raw_record_edges(self):
        for hops in (1, 2, 4, 7):
            spec = TaskSpec(hops=hops, length=80, min_length=48, query_gap=3)
            for example in build_split(spec, "validation", 13, 32).examples:
                table = {example.tokens[i + 1]: example.tokens[i + 3]
                         for i, token in enumerate(example.tokens) if token == v.KEY}
                end = example.tokens.index(v.END_TABLE)
                self.assertEqual(len(table), spec.pairs)
                for position, target in enumerate(example.targets):
                    if target == v.IGNORE:
                        continue
                    self.assertGreater(position, end + spec.query_gap)
                    current = example.tokens[position]
                    visited = {current}
                    for _ in range(hops):
                        current = table[current]
                        self.assertNotIn(current, visited)
                        visited.add(current)
                    self.assertEqual(current, target)

    def test_matched_lookup_changes_answers_with_identical_counts_and_queries(self):
        for seed in range(20):
            spec = TaskSpec()
            first, second = lookup_pair(spec, random.Random(seed), 64)
            self.assertEqual(Counter(first.tokens), Counter(second.tokens))
            self.assertNotEqual(first.targets, second.targets)
            for index, target in enumerate(first.targets):
                if target != v.IGNORE:
                    self.assertEqual(first.tokens[index], second.tokens[index])
            validate_example(first, spec)
            validate_example(second, spec)

    def test_prefix_pairs_preserve_counts_endpoints_and_neutral_positions(self):
        for task in ("dyck", "blocks"):
            for seed in range(20):
                spec = TaskSpec(task=task, max_neutral_gap=4, max_balance=3)
                first, second = prefix_pair(spec, random.Random(seed), 64)
                self.assertEqual(Counter(first.tokens), Counter(second.tokens))
                self.assertEqual(first.tokens[0], second.tokens[0])
                self.assertEqual(first.tokens[-1], second.tokens[-1])
                self.assertEqual([i for i, x in enumerate(first.tokens) if x == v.NEUTRAL],
                                 [i for i, x in enumerate(second.tokens) if x == v.NEUTRAL])
                self.assertNotEqual(first.targets[-1], second.targets[-1])

    def test_neutral_cap_and_balance_controls_are_respected(self):
        for maximum in (1, 2, 4):
            spec = TaskSpec(task="dyck", max_balance=maximum, max_neutral_gap=2, neutral_fraction=0.4)
            for example in build_split(spec, "train", 15, 64).examples:
                run = balance = 0
                for token in example.tokens[1:]:
                    run = run + 1 if token == v.NEUTRAL else 0
                    self.assertLessEqual(run, 2)
                    balance += (token == v.OPEN) - (token == v.CLOSE)
                    self.assertLessEqual(abs(balance), maximum)

    def test_background_examples_expose_wrong_start_and_other_block_counts(self):
        spec = TaskSpec(task="blocks")
        examples = build_split(spec, "train", 8, 128).examples
        active = [[x for x in example.tokens[1:] if x != v.NEUTRAL] for example in examples]
        self.assertIn(v.B, [word[0] for word in active])
        runs = {sum(i == 0 or token != word[i - 1] for i, token in enumerate(word)) for word in active}
        self.assertTrue({1, 2, 3, 4, 5}.issubset(runs))

    def test_variable_lengths_and_all_initial_depths_generate_valid_examples(self):
        for task in TASKS:
            for depth in (1, 2, 3, 4):
                spec = TaskSpec(task=task, hops=depth, blocks=depth, length=80, min_length=48)
                examples = build_split(spec, "train", 3, 32).examples
                self.assertGreater(len({len(example.tokens) for example in examples}), 1)
                for example in examples:
                    validate_example(example, spec)

    def test_prefix_output_labels_are_absent_from_model_inputs(self):
        labels = {v.BALANCED, v.INCOMPLETE, v.INVALID, v.REJECT, v.ACCEPT}
        for task in ("dyck", "blocks"):
            examples = build_split(TaskSpec(task=task), "train", 0, 32).examples
            self.assertTrue(all(labels.isdisjoint(example.tokens) for example in examples))

    def test_splits_are_reproducible_and_have_independent_seed_streams(self):
        for task in TASKS:
            spec = TaskSpec(task=task)
            first = build_split(spec, "train", 17, 16)
            self.assertEqual(first, build_split(spec, "train", 17, 16))
            others = [build_split(spec, name, seed, 16) for name, seed in
                      (("validation", 17), ("test", 17), ("train", 18))]
            for other in others:
                self.assertNotEqual(first.fingerprint, other.fingerprint)
                self.assertNotEqual(first.examples, other.examples)
            changed = build_split(replace(spec, length=128, min_length=None), "test", 17, 16)
            self.assertNotEqual(first.fingerprint, changed.fingerprint)

    def test_saved_data_and_fingerprint_match_observed_examples(self):
        split = build_split(TaskSpec(), "train", 0, 5)
        with tempfile.TemporaryDirectory() as root:
            split.save(root)
            metadata = json.loads((Path(root) / "metadata.json").read_text())
            rows = [json.loads(line) for line in (Path(root) / "examples.jsonl").read_text().splitlines()]
            self.assertEqual(metadata["fingerprint"], split.fingerprint)
            self.assertEqual(metadata["count"], len(rows))
            self.assertEqual(rows[0]["tokens"], list(split.examples[0].tokens))
            self.assertEqual(rows[0]["targets"], list(split.examples[0].targets))

    def test_unrelated_task_controls_do_not_change_reference_splits(self):
        for task, fields in (("mqar", dict(hops=4, blocks=4, max_balance=1)),
                             ("lookup", dict(alpha=2, blocks=4)),
                             ("dyck", dict(symbols=64, blocks=4)),
                             ("blocks", dict(symbols=64, max_balance=1))):
            spec = TaskSpec(task=task)
            original = build_split(spec, "train", 0, 16)
            changed = build_split(replace(spec, **fields), "train", 0, 16)
            self.assertEqual(original.examples, changed.examples)
            self.assertEqual(original.fingerprint, changed.fingerprint)

    def test_invalid_configurations_fail_instead_of_hanging_or_leaking_answers(self):
        cases = (dict(task="missing"), dict(length=0), dict(min_length=100),
                 dict(pairs=2, hops=2), dict(queries=9), dict(symbols=2), dict(query_gap=99),
                 dict(alpha=float("nan")), dict(task="dyck", neutral_fraction=1),
                 dict(task="dyck", max_balance=0), dict(task="blocks", blocks=0),
                 dict(task="blocks", neutral_fraction=0.99),
                 dict(task="dyck", length=9, min_length=7, neutral_fraction=0, max_neutral_gap=0))
        for fields in cases:
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                TaskSpec(**fields)
        with self.assertRaises(ValueError):
            build_split(TaskSpec(), "test", 0, 0)
