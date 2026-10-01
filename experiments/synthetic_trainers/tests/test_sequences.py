"""Sequence oracles checked by exhaustive, independent elementary definitions."""

from dataclasses import replace
import itertools
import unittest

from experiments.synthetic_trainers import vocabulary as v
from experiments.synthetic_trainers.data import build_split
from experiments.synthetic_trainers.oracles import validate_example
from experiments.synthetic_trainers.records import generation_example
from experiments.synthetic_trainers.sequence_oracles import generation_answer, generation_limit
from experiments.synthetic_trainers.specs import TaskSpec


def prompt_for(spec, word):
    bos = spec.histogram_bos if spec.task == "histogram" else True
    return (*([v.BOS] if bos else []), *word, v.SEP)


def short_words():
    for size in range(1, 6):
        yield from itertools.product(range(v.IDENTITY_BASE, v.IDENTITY_BASE + 3), repeat=size)


class SequenceTests(unittest.TestCase):
    def spec(self, task, **fields):
        return TaskSpec(task=task, length=6, symbols=8, number_limit=32, **fields)

    def test_copy_every_short_word_preserves_order_and_multiplicity(self):
        spec = self.spec("copy")
        for word in short_words():
            self.assertEqual(generation_answer(prompt_for(spec, word), spec), (*word, v.EOS))

    def test_reverse_every_short_word_including_repeated_symbols(self):
        spec = self.spec("reverse")
        for word in short_words():
            expected = tuple(word[len(word) - 1 - i] for i in range(len(word)))
            self.assertEqual(generation_answer(prompt_for(spec, word), spec), (*expected, v.EOS))

    def test_sort_every_short_word_by_independent_insertion_sort(self):
        spec = self.spec("sort")
        for word in short_words():
            expected = []
            for token in word:
                index = 0
                while index < len(expected) and expected[index] <= token:
                    index += 1
                expected.insert(index, token)
            self.assertEqual(generation_answer(prompt_for(spec, word), spec), (*expected, v.EOS))

    def test_histogram_with_and_without_bos_counts_only_data_symbols(self):
        for bos in (True, False):
            spec = self.spec("histogram", histogram_bos=bos)
            for word in short_words():
                numbers = [spec.number_base + sum(other == token for other in word) for token in word]
                expected = (*([v.BOS] if bos else []), *numbers, v.EOS)
                self.assertEqual(generation_answer(prompt_for(spec, word), spec), expected)

    def test_double_histogram_counts_types_rather_than_positions(self):
        spec = self.spec("histogram2")
        for word in short_words():
            expected = [v.BOS]
            for token in word:
                frequency = word.count(token)
                types = sum(word.count(other) == frequency for other in set(word))
                expected.append(spec.number_base + types)
            self.assertEqual(generation_answer(prompt_for(spec, word), spec), (*expected, v.EOS))

    def test_double_histogram_matches_paper_abbc_example(self):
        spec = self.spec("histogram2")
        a, b, c = range(v.IDENTITY_BASE, v.IDENTITY_BASE + 3)
        self.assertEqual(generation_answer(prompt_for(spec, (a, b, b, c)), spec),
                         (v.BOS, *(spec.number_base + n for n in (2, 1, 1, 2)), v.EOS))

    def test_mode_every_short_word_with_a_unique_maximum(self):
        spec = self.spec("mode")
        for word in short_words():
            best = max(word.count(token) for token in word)
            winners = [token for token in set(word) if word.count(token) == best]
            if len(winners) == 1:
                self.assertEqual(generation_answer(prompt_for(spec, word), spec), (winners[0], v.EOS))
            else:
                with self.assertRaisesRegex(ValueError, "unique"):
                    generation_answer(prompt_for(spec, word), spec)

    def test_most_frequent_ties_follow_first_occurrence_and_use_bos_padding(self):
        spec = self.spec("most_freq")
        a, b, c = range(v.IDENTITY_BASE, v.IDENTITY_BASE + 3)
        word = (c, a, b, b, a, c)
        self.assertEqual(generation_answer(prompt_for(spec, word), spec),
                         (v.BOS, c, a, b, v.BOS, v.BOS, v.BOS, v.EOS))

    def test_most_frequent_every_short_word_returns_each_type_once(self):
        spec = self.spec("most_freq")
        for word in short_words():
            remaining, expected = list(dict.fromkeys(word)), []
            while remaining:
                best = max(word.count(x) for x in remaining)
                chosen = next(x for x in remaining if word.count(x) == best)
                expected.append(chosen)
                remaining.remove(chosen)
            expected += [v.BOS] * (len(word) - len(expected))
            self.assertEqual(generation_answer(prompt_for(spec, word), spec), (v.BOS, *expected, v.EOS))

    def test_mode_scratchpad_variants_keep_the_same_final_answer(self):
        a, b, c = range(v.IDENTITY_BASE, v.IDENTITY_BASE + 3)
        word = (b, a, c, a, b, a)
        for mode, order in (("counts", (c, b, a)), ("itemized", (b, a, c))):
            spec = self.spec("mode", scratchpad=mode)
            expected = []
            for token in order:
                expected.extend((spec.number_base + word.count(token), token))
            self.assertEqual(generation_answer(prompt_for(spec, word), spec), (*expected, a, v.EOS))

    def test_atomic_count_includes_both_endpoints_and_singleton_intervals(self):
        spec = self.spec("count")
        for start in range(1, 12):
            for end in range(start, start + 5):
                prompt = (v.BOS, spec.number_base + start, spec.number_base + end, v.SEP)
                expected = tuple(spec.number_base + start + i for i in range(end - start + 1))
                self.assertEqual(generation_answer(prompt, spec), (*expected, v.EOS))
                self.assertEqual(generation_limit(prompt, spec), len(expected) + 1)

    def test_count_rejects_descending_noninteger_or_incomplete_prompts(self):
        spec = self.spec("count")
        for body in ((spec.number_base + 3, spec.number_base + 2), (v.A, v.B),
                     (spec.number_base + 1,), (spec.number_base, spec.number_base + 1)):
            with self.assertRaises(ValueError):
                generation_answer((v.BOS, *body, v.SEP), spec)

    def test_unique_token_variants_reject_repeats(self):
        for task in ("copy", "reverse", "sort"):
            spec = self.spec(task, unique=True)
            with self.assertRaisesRegex(ValueError, "repeated"):
                generation_answer(prompt_for(spec, (v.IDENTITY_BASE, v.IDENTITY_BASE)), spec)
            for row in build_split(spec, "train", 2, 16).examples:
                word = row.prompt[1:-1]
                self.assertEqual(len(set(word)), len(word))

    def test_generation_target_is_always_one_token_ahead_of_visible_answers(self):
        for task in ("copy", "reverse", "sort", "histogram", "histogram2", "mode", "most_freq", "count"):
            spec = self.spec(task)
            for row in build_split(spec, "train", 5, 16).examples:
                for index, target in enumerate(row.answer):
                    position = len(row.prompt) - 1 + index
                    self.assertEqual(row.tokens[:position + 1], row.prompt + row.answer[:index])
                    self.assertEqual(row.targets[position], target)
                self.assertEqual(row.targets[:len(row.prompt) - 1], (v.IGNORE,) * (len(row.prompt) - 1))
                validate_example(row, spec)

    def test_validator_rejects_answer_corruption_and_reference_based_limits(self):
        spec = self.spec("mode", scratchpad="counts")
        row = build_split(spec, "test", 0, 1).examples[0]
        wrong = next(token for token in range(v.IDENTITY_BASE, v.IDENTITY_BASE + spec.symbols) if token != row.answer[-2])
        corrupted = (*row.answer[:-2], wrong, v.EOS)
        example = generation_example(spec.task, row.prompt, corrupted, row.generation_limit)
        with self.assertRaises(ValueError):
            validate_example(example, spec)
        with self.assertRaises(ValueError):
            validate_example(replace(row, generation_limit=row.generation_limit + 1), spec)

    def test_empty_bad_alphabet_and_missing_separator_fail_before_training(self):
        spec = self.spec("copy")
        for prompt in ((v.BOS, v.SEP), (v.BOS, v.ACCEPT, v.SEP),
                       (v.BOS, v.IDENTITY_BASE), (v.BOS, v.SEP, v.IDENTITY_BASE, v.SEP)):
            with self.assertRaises(ValueError):
                generation_answer(prompt, spec)
