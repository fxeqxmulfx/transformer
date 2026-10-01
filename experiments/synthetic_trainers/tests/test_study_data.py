"""Finite datasets and corruption cannot leak answers or change across repeats."""

from dataclasses import replace
import math
import unittest

from experiments.synthetic_trainers.config import TrainConfig
from experiments.synthetic_trainers.corpus import context_report, corpus_report, study_pool
from experiments.synthetic_trainers.data import build_split
from experiments.synthetic_trainers.label_noise import corrupt_example, label_domain, noisy_split
from experiments.synthetic_trainers.oracles import oracle_targets, validate_example
from experiments.synthetic_trainers.records import example_from_targets
from experiments.synthetic_trainers.specs import TASKS, TaskSpec
from experiments.synthetic_trainers.suite import expand_variants
from experiments.synthetic_trainers import vocabulary as v


class StudyDataTests(unittest.TestCase):
    def test_random_control_is_repeatable_nested_and_has_no_prompt_oracle(self):
        spec = TaskSpec(task="random_lm", length=5, min_length=2, symbols=4, number_limit=16)
        short = build_split(spec, "train", 4, 7)
        long = build_split(spec, "train", 4, 50)
        self.assertEqual(short.examples, long.examples[:7])
        self.assertEqual(short, build_split(spec, "train", 4, 7))
        self.assertNotEqual(long.examples, build_split(spec, "validation", 4, 50).examples)
        self.assertGreater(len({row.answer for row in long.examples}), 1)
        for row in long.examples:
            validate_example(row, spec)
            self.assertEqual(row.prompt, (v.BOS, spec.number_base + len(row.answer) - 1, v.SEP))
            self.assertEqual(row.tokens, row.prompt + row.answer[:-1])
            self.assertEqual(row.generation_limit, len(row.answer))
            with self.assertRaisesRegex(ValueError, "no deterministic"):
                oracle_targets(row.tokens, spec)
        report = context_report(long.examples)
        self.assertGreater(report["conflicting_contexts"], 0)
        self.assertGreater(report["empirical_min_token_loss_nats"], 0)

    def test_invalid_random_payload_and_ood_numeric_capacity_fail(self):
        spec = TaskSpec(task="random_lm", length=3, symbols=2, number_limit=4)
        row = build_split(spec, "train", 0, 1).examples[0]
        from experiments.synthetic_trainers.records import generation_example

        bad = generation_example(spec.task, row.prompt, (v.ONE, *row.answer[1:]), 4)
        with self.assertRaises(ValueError):
            validate_example(bad, spec)
        with self.assertRaises(ValueError):
            replace(spec, length=8)

    def test_all_37_formats_accept_frozen_noise_and_preserve_structural_targets(self):
        variants = 0
        for task in TASKS:
            base = TaskSpec(task=task, length=24, symbols=48, pairs=4, queries=2, number_limit=64)
            for spec in expand_variants(base):
                variants += 1
                clean = build_split(spec, "train", 3, 4)
                observed, info = noisy_split(clean, 1.0, 5)
                self.assertEqual(info["changed_targets"], info["eligible_targets"])
                self.assertGreater(info["changed_targets"], 0)
                self.assertEqual(observed, noisy_split(clean, 1.0, 5)[0])
                for original, noisy in zip(clean.examples, observed.examples):
                    validate_example(original, spec)
                    for i, target in enumerate(original.targets):
                        domain = label_domain(original, i, spec)
                        self.assertEqual(noisy.targets[i] == target, len(domain) < 2)
                        if domain:
                            self.assertIn(noisy.targets[i], domain)
                    if spec.generative:
                        self.assertEqual(noisy.prompt, original.prompt)
                        self.assertEqual(noisy.answer[-1], v.EOS)
                        self.assertEqual(noisy.tokens, noisy.prompt + noisy.answer[:-1])
                        self.assertEqual(noisy.targets[len(noisy.prompt) - 1:], noisy.answer)
                    else:
                        self.assertEqual(noisy.tokens, original.tokens)
                self.assertEqual(clean, build_split(spec, "train", 3, 4))
        self.assertEqual(variants, 37)

    def test_noise_assignments_are_nested_by_rate_and_stable_by_pool_size(self):
        spec = TaskSpec(task="copy", length=4, symbols=8)
        clean = build_split(spec, "train", 0, 5)
        a, _ = noisy_split(clean, 0.25, 8)
        b, _ = noisy_split(clean, 0.75, 8)
        bigger, _ = noisy_split(build_split(spec, "train", 0, 8), 0.25, 8)
        self.assertEqual(a.examples, bigger.examples[:5])
        for original, first, second in zip(clean.examples, a.examples, b.examples):
            for right, low, high in zip(original.targets, first.targets, second.targets):
                if low != right:
                    self.assertEqual(low, high)
        duplicated = replace(clean, examples=(clean.examples[0], clean.examples[0]))
        self.assertEqual(noisy_split(duplicated, 0.5, 3)[0].examples[0], noisy_split(duplicated, 0.5, 3)[0].examples[1])
        self.assertEqual(noisy_split(clean, 0.0, 100)[0], clean)

    def test_incorrect_label_kernel_has_the_requested_rate_and_uniform_alternatives(self):
        spec = TaskSpec(task="dyck", length=24)
        counts = {v.INCOMPLETE: 0, v.INVALID: 0}
        total = flipped = 0
        for identity in range(1500):
            # Different causal addresses exercise the independent keyed draws.
            row = example_from_targets("dyck", (v.BOS, identity), (v.IGNORE, v.BALANCED))
            result, eligible, changed = corrupt_example(row, spec, 0.4, 0)
            total += eligible
            flipped += changed
            if changed:
                counts[result.targets[-1]] += 1
        self.assertAlmostEqual(flipped / total, 0.4, delta=0.04)
        self.assertAlmostEqual(counts[v.INVALID] / flipped, 0.5, delta=0.07)

    def test_disjoint_pool_and_independent_overlap_diagnostics(self):
        spec = TaskSpec(task="copy", length=4, symbols=8)
        config = TrainConfig(study="memorization", split_policy="disjoint", train_examples=9, validation_examples=5, test_examples=5)
        pool = study_pool(spec, config)
        self.assertTrue(all(item["unique_inputs"] == 0 for item in corpus_report(pool)["overlaps"].values()))
        larger = study_pool(spec, replace(config, train_examples=12))
        self.assertEqual(pool["train"].examples, larger["train"].examples[:9])
        duplicate = replace(pool["validation"], examples=(pool["train"].examples[0],) * 3)
        report = corpus_report({"train": pool["train"], "validation": duplicate})
        self.assertEqual(report["splits"]["validation"]["duplicate_rows"], 2)
        self.assertEqual(report["overlaps"]["train/validation"]["rows_in_second"], 3)

    def test_exhausted_domain_and_non_iid_random_policy_fail_clearly(self):
        config = TrainConfig(study="double_descent", split_policy="disjoint", train_examples=2, validation_examples=1, test_examples=1)
        with self.assertRaisesRegex(ValueError, "disjoint"):
            study_pool(TaskSpec(task="boolean_and", length=4), replace(config, train_examples=8))
        with self.assertRaisesRegex(ValueError, "independent"):
            study_pool(TaskSpec(task="random_lm", length=4), config)

    def test_conflict_floor_is_the_empirical_conditional_entropy(self):
        examples = [example_from_targets("lookup", (v.BOS,), (label,)) for label in (v.ACCEPT, v.ACCEPT, v.REJECT)]
        report = context_report(examples)
        self.assertEqual(report["conflicting_contexts"], 1)
        self.assertAlmostEqual(report["empirical_min_token_error"], 1 / 3)
        self.assertAlmostEqual(report["empirical_min_token_loss_nats"], -(2 * math.log(2 / 3) + math.log(1 / 3)) / 3)

    def test_invalid_study_controls_are_rejected(self):
        for fields in ({"label_noise": 0.1}, {"study": "double_descent", "stop_at_target": True},
                       {"study": "memorization", "label_noise": float("nan")}, {"fit_epsilon": 0},
                       {"generalization_patience": 0}, {"curve_tolerance": -1}, {"noise_seed": -1}):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                TrainConfig(**fields)
