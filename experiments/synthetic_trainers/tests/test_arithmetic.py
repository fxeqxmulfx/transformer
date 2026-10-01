"""Carries, index correspondence, parity state, and positional distribution shift."""

from dataclasses import replace
import itertools
import random
import unittest

from experiments.synthetic_trainers import vocabulary as v
from experiments.synthetic_trainers.arithmetic import addition_inputs, addition_prompt, carry_profile, longest_carry
from experiments.synthetic_trainers.data import build_split
from experiments.synthetic_trainers.sequence_oracles import bit_inputs, generation_answer, generation_limit
from experiments.synthetic_trainers.specs import TaskSpec


def decimal_prompt(spec, a, b, size, hint_start=3):
    left, right = str(a).zfill(size + 1), str(b).zfill(size + 1)
    prompt = [v.BOS]
    for offset, digits in enumerate((left, right)):
        if offset:
            prompt.append(v.PLUS)
        for index, digit in enumerate(digits):
            if spec.index_hints:
                prompt.append(spec.number_base + hint_start + index)
            prompt.append(v.DIGIT_BASE + int(digit))
    return (*prompt, v.SEP)


def binary_prompt(spec, bits):
    body = []
    for index, bit in enumerate(bits):
        if spec.index_hints:
            body.append(spec.number_base + 2 + index)
        body.append(v.ONE if bit else v.ZERO)
    return (v.BOS, *body, v.SEP)


class ArithmeticTests(unittest.TestCase):
    def test_addition_all_small_pairs_forward_and_reverse_with_and_without_hints(self):
        for order, hints in itertools.product(("forward", "reverse"), (False, True)):
            spec = TaskSpec(task="addition", length=2, symbols=4, number_limit=32,
                            addition_order=order, index_hints=hints)
            for a, b in itertools.product(range(40), repeat=2):
                prompt = decimal_prompt(spec, a, b, 2)
                answer = generation_answer(prompt, spec)
                digits = answer[1:-1:2] if hints else answer[:-1]
                indices = answer[:-1:2] if hints else ()
                text = "".join(str(digit - v.DIGIT_BASE) for digit in digits)
                self.assertEqual(int(text if order == "forward" else text[::-1]), a + b)
                if hints:
                    expected = tuple(spec.number_base + 3 + i for i in range(3))
                    self.assertEqual(indices, expected if order == "forward" else expected[::-1])
                self.assertEqual(answer[-1], v.EOS)

    def test_hard_carry_paper_example_381_plus_619(self):
        spec = TaskSpec(task="addition", length=3)
        answer = generation_answer(decimal_prompt(spec, 381, 619, 3), spec)
        self.assertEqual(answer, (v.DIGIT_BASE + 1, v.DIGIT_BASE, v.DIGIT_BASE, v.DIGIT_BASE, v.EOS))
        self.assertEqual(carry_profile((0, 3, 8, 1), (0, 6, 1, 9)), (1, 1, 1, 0))

    def test_arbitrarily_long_carry_oracle_uses_exact_integer_arithmetic(self):
        size = 80
        for order in ("forward", "reverse"):
            spec = TaskSpec(task="addition", length=size, addition_order=order)
            answer = generation_answer(decimal_prompt(spec, 10 ** size - 1, 1, size), spec)
            expected = [v.DIGIT_BASE + 1] + [v.DIGIT_BASE] * size
            self.assertEqual(answer, (*(expected if order == "forward" else expected[::-1]), v.EOS))

    def test_sampler_constructs_each_requested_carry_chain_exactly(self):
        for size in range(1, 13):
            for chain in range(size + 1):
                spec = TaskSpec(task="addition", length=size, carry_length=chain)
                for seed in range(12):
                    prompt = addition_prompt(spec, random.Random(seed), size)
                    left, right, _ = addition_inputs(prompt, spec)
                    profile = carry_profile(left, right)
                    self.assertEqual(longest_carry(left, right), chain)
                    self.assertEqual(sum(profile), chain)
                    self.assertEqual(profile[-1], 0)  # Extra leading padding digit absorbs overflow.

    def test_balanced_sampling_exposes_all_chain_lengths(self):
        spec = TaskSpec(task="addition", length=12)
        examples = build_split(spec, "train", 7, 256).examples
        lengths = {longest_carry(*addition_inputs(row.prompt, spec)[:2]) for row in examples}
        self.assertEqual(lengths, set(range(13)))

    def test_carry_length_varies_independently_of_operand_width(self):
        spec = TaskSpec(task="addition", length=12, min_length=4, carry_length=3)
        rows = build_split(spec, "train", 13, 64).examples
        widths = set()
        for row in rows:
            left, right, _ = addition_inputs(row.prompt, spec)
            widths.add(len(left))
            self.assertEqual(longest_carry(left, right), 3)
        self.assertGreater(len(widths), 1)

    def test_addition_rejects_misaligned_hints_and_invalid_digits(self):
        spec = TaskSpec(task="addition", length=3, index_hints=True)
        correct = list(decimal_prompt(spec, 381, 619, 3))
        split = correct.index(v.PLUS)
        cases = []
        for index, token in ((1, correct[1] + 2), (2, v.A), (split + 1, correct[1] + 1)):
            corrupted = correct.copy()
            corrupted[index] = token
            cases.append(corrupted)
        cases.append(correct[:-1])
        for prompt in cases:
            with self.assertRaises(ValueError):
                generation_answer(prompt, spec)

    def test_parity_all_short_bit_strings_in_every_scratchpad_format(self):
        formats = (("none", False), ("running", False), ("running", True), ("ones", True))
        for mode, hints in formats:
            spec = TaskSpec(task="parity", length=6, scratchpad=mode, index_hints=hints)
            for size in range(1, 7):
                for bits in itertools.product((0, 1), repeat=size):
                    answer = generation_answer(binary_prompt(spec, bits), spec)
                    self.assertEqual(answer[-2], v.ODD if sum(bits) % 2 else v.EVEN)
                    self.assertEqual(answer[-1], v.EOS)
                    if mode != "none":
                        expected = [v.EVEN]
                        for index in range(size):
                            if mode == "ones" and not bits[index]:
                                continue
                            if hints:
                                expected.append(spec.number_base + 2 + index)
                            expected.append(v.ODD if sum(bits[:index + 1]) % 2 else v.EVEN)
                        self.assertEqual(answer, (*expected, v.EOS))

    def test_parity_zero_word_scratchpad_terminates_without_fake_steps(self):
        spec = TaskSpec(task="parity", length=6, scratchpad="ones", index_hints=True)
        prompt = binary_prompt(spec, [0] * 6)
        self.assertEqual(generation_answer(prompt, spec), (v.EVEN, v.EOS))
        self.assertEqual(generation_limit(prompt, spec), 14)
        self.assertEqual(generation_limit(binary_prompt(spec, [1] * 6), spec), 14)

    def test_parity_rejects_nondisjoint_bits_and_broken_hint_order(self):
        spec = TaskSpec(task="parity", length=6, index_hints=True)
        for prompt in ((v.BOS, v.A, v.SEP),
                       (v.BOS, spec.number_base + 2, v.ONE, spec.number_base + 1, v.ZERO, v.SEP)):
            with self.assertRaises(ValueError):
                generation_answer(prompt, spec)

    def test_boolean_and_all_short_words_detects_a_zero_in_every_position(self):
        spec = TaskSpec(task="boolean_and", length=6)
        for bits in itertools.product((0, 1), repeat=6):
            prompt = (v.BOS, *(v.ONE if bit else v.ZERO for bit in bits), v.SEP)
            self.assertEqual(generation_answer(prompt, spec),
                             (v.ACCEPT if sum(bits) == 6 else v.REJECT, v.EOS))

    def test_boolean_and_train_and_shifted_test_have_disjoint_zero_positions(self):
        spec = TaskSpec(task="boolean_and", length=20)
        early = build_split(spec, "train", 0, 256)
        late = build_split(replace(spec, and_region="late"), "test", 0, 256)
        regions = []
        for split in (early, late):
            positions, labels = set(), set()
            for row in split.examples:
                word = row.prompt[1:-1]
                self.assertLessEqual(word.count(v.ZERO), 1)
                positions.update(i for i, token in enumerate(word) if token == v.ZERO)
                labels.add(row.answer[0])
            self.assertEqual(labels, {v.ACCEPT, v.REJECT})
            regions.append(positions)
        self.assertEqual(regions[0], set(range(15)))
        self.assertEqual(regions[1], set(range(15, 20)))
        self.assertTrue(regions[0].isdisjoint(regions[1]))

    def test_boolean_and_random_control_covers_the_entire_position_range(self):
        spec = TaskSpec(task="boolean_and", length=12, and_shift=False)
        rows = build_split(spec, "train", 6, 256).examples
        positions = {i for row in rows for i, token in enumerate(row.prompt[1:-1]) if token == v.ZERO}
        self.assertEqual(positions, set(range(12)))
