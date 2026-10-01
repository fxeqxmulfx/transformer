"""Complete variant matrix, paired input problems, reproducibility, and budgets."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from experiments.synthetic_trainers import vocabulary as v
from experiments.synthetic_trainers.arithmetic import addition_inputs
from experiments.synthetic_trainers.data import GENERATOR_VERSION, build_split
from experiments.synthetic_trainers.oracles import validate_example
from experiments.synthetic_trainers.sequence_oracles import bit_inputs
from experiments.synthetic_trainers.specs import TASKS, TaskSpec
from experiments.synthetic_trainers.suite import expand_variants, task_names


class SuiteDataTests(unittest.TestCase):
    def test_every_variant_supports_variable_lengths_and_ood_with_exact_oracle_labels(self):
        names = set()
        for task in TASKS:
            spec = TaskSpec(task=task, length=24, min_length=22, symbols=96,
                            number_limit=128, pairs=4, queries=2)
            for variant in expand_variants(spec):
                with self.subTest(variant=variant.run_name):
                    self.assertNotIn(variant.run_name, names)
                    names.add(variant.run_name)
                    for configured in (variant, replace(variant, length=48, min_length=None)):
                        rows = build_split(configured, "test", 4, 8).examples
                        for row in rows:
                            validate_example(row, configured)
                            self.assertLessEqual(len(row.tokens), configured.context_length)
                            self.assertTrue(all(0 <= token < configured.vocab_size
                                                for token in (*row.tokens, *(x for x in row.targets if x != v.IGNORE))))
        self.assertEqual(len(names), 37)

    def test_each_variant_has_reproducible_fingerprints_and_independent_split_streams(self):
        for task in TASKS:
            spec = TaskSpec(task=task, length=24, symbols=48, number_limit=64, pairs=4, queries=2)
            for variant in expand_variants(spec):
                first = build_split(variant, "train", 5, 5)
                self.assertEqual(first.fingerprint, build_split(variant, "train", 5, 5).fingerprint)
                self.assertNotEqual(first.fingerprint, build_split(variant, "test", 5, 5).fingerprint)
                self.assertNotEqual(first.fingerprint, build_split(variant, "train", 6, 5).fingerprint)

    def test_addition_formats_share_exact_operand_pairs(self):
        spec = TaskSpec(task="addition", length=12, min_length=4, symbols=8, number_limit=64)
        baseline = build_split(spec, "train", 7, 32)
        operands = [addition_inputs(row.prompt, spec)[:2] for row in baseline.examples]
        for order, hints in (("reverse", False), ("forward", True), ("reverse", True)):
            variant = replace(spec, addition_order=order, index_hints=hints)
            changed = build_split(variant, "train", 7, 32)
            self.assertEqual([addition_inputs(row.prompt, variant)[:2] for row in changed.examples], operands)
            self.assertNotEqual(changed.fingerprint, baseline.fingerprint)

    def test_parity_scratchpads_and_hints_share_exact_input_bits(self):
        spec = TaskSpec(task="parity", length=12, min_length=3, symbols=8, number_limit=64)
        baseline = build_split(spec, "train", 3, 32)
        bits = [bit_inputs(row.prompt, spec)[0] for row in baseline.examples]
        for variant in expand_variants(spec):
            changed = build_split(variant, "train", 3, 32)
            self.assertEqual([bit_inputs(row.prompt, variant)[0] for row in changed.examples], bits)

    def test_mode_scratchpad_variants_share_inputs_and_final_answers(self):
        spec = TaskSpec(task="mode", length=12, min_length=3, symbols=8, number_limit=32)
        variants = [build_split(variant, "train", 8, 32) for variant in expand_variants(spec)]
        for changed in variants[1:]:
            self.assertEqual([row.prompt for row in changed.examples], [row.prompt for row in variants[0].examples])
            self.assertEqual([row.answer[-2] for row in changed.examples], [row.answer[-2] for row in variants[0].examples])

    def test_histogram_bos_control_preserves_the_underlying_word(self):
        spec = TaskSpec(task="histogram", length=12, min_length=2, symbols=8, number_limit=32)
        with_bos = build_split(spec, "train", 4, 16)
        without_bos = build_split(replace(spec, histogram_bos=False), "train", 4, 16)
        self.assertEqual([row.prompt[1:] for row in with_bos.examples], [row.prompt for row in without_bos.examples])
        self.assertEqual([row.answer[1:] for row in with_bos.examples], [row.answer for row in without_bos.examples])

    def test_unrelated_controls_do_not_change_any_new_tasks_data(self):
        for task in TASKS[4:]:
            spec = TaskSpec(task=task, length=12, symbols=16, number_limit=32)
            changed = replace(spec, pairs=99, queries=77, hops=66, alpha=5)
            self.assertEqual(build_split(spec, "train", 0, 8).fingerprint,
                             build_split(changed, "train", 0, 8).fingerprint)

    def test_number_and_identity_namespaces_are_disjoint_at_ood_lengths(self):
        for task in ("histogram", "histogram2", "count", "addition", "parity", "mode"):
            fields = {"index_hints": True} if task in ("addition", "parity") else {"scratchpad": "counts"} if task == "mode" else {}
            spec = TaskSpec(task=task, length=8, symbols=16, number_limit=64, **fields)
            ood = replace(spec, length=32)
            self.assertEqual(spec.vocab_size, ood.vocab_size)
            self.assertEqual(spec.number_base, v.IDENTITY_BASE + spec.symbols)
            for row in build_split(ood, "test", 0, 8).examples:
                self.assertLess(max(row.tokens), spec.vocab_size)
                self.assertLess(max(row.answer), spec.vocab_size)

    def test_saved_generation_examples_include_prompt_answer_and_independent_limit(self):
        spec = TaskSpec(task="parity", length=8, scratchpad="ones", index_hints=True)
        split = build_split(spec, "test", 6, 3)
        with tempfile.TemporaryDirectory() as root:
            split.save(root)
            metadata = json.loads((Path(root) / "metadata.json").read_text())
            rows = [json.loads(line) for line in (Path(root) / "examples.jsonl").read_text().splitlines()]
            self.assertEqual(metadata["version"], GENERATOR_VERSION)
            for row, example in zip(rows, split.examples):
                self.assertEqual(row["prompt"], list(example.prompt))
                self.assertEqual(row["answer"], list(example.answer))
                self.assertEqual(row["generation_limit"], example.generation_limit)

    def test_invalid_new_difficulty_controls_fail_early(self):
        cases = (dict(task="copy", length=8, symbols=4, unique=True),
                 dict(task="histogram", length=12, number_limit=8),
                 dict(task="mode", length=2), dict(task="mode", scratchpad="running"),
                 dict(task="parity", scratchpad="counts"),
                 dict(task="addition", addition_order="sideways"),
                 dict(task="addition", carry_sampling="random"),
                 dict(task="addition", length=8, min_length=3, carry_length=4),
                 dict(task="addition", length=8, number_limit=8, index_hints=True),
                 dict(task="boolean_and", length=3), dict(task="boolean_and", and_region="middle"),
                 dict(task="dyck2", bracket_types=1), dict(task="dyck2", bracket_types=9),
                 dict(task="crasp", formula_depth=0), dict(task="crasp", formula_depth=6),
                 dict(task="crasp", formula_seed=-1))
        for fields in cases:
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                TaskSpec(**fields)

    def test_group_selectors_cover_paper_task_lists_without_duplicates(self):
        self.assertEqual(set(task_names("rasp")),
                         {"histogram", "histogram2", "mode", "most_freq", "copy", "reverse", "sort", "dyck2"})
        self.assertEqual(set(task_names("rasp-l")), {"count", "mode", "copy", "sort", "addition", "parity", "boolean_and"})
        self.assertEqual(len(task_names("all")), len(set(TASKS)))
        self.assertEqual(task_names("prefix", "crasp"), ("crasp",))
