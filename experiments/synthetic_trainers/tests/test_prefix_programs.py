"""Typed stacks and temporal formulas checked independently on finite words."""

from collections import Counter
from dataclasses import asdict
import itertools
import json
import random
import unittest

from experiments.synthetic_trainers import vocabulary as v
from experiments.synthetic_trainers.crasp import Node, crasp_targets, evaluate_formula, program_with_witnesses
from experiments.synthetic_trainers.data import build_split
from experiments.synthetic_trainers.specs import TaskSpec
from experiments.synthetic_trainers.typed_dyck import typed_pair, typed_targets


def typed_stack_status(word, pairs):
    stack = []
    for token in word:
        if token == v.NEUTRAL:
            continue
        openings = [opening for opening, _ in pairs]
        closings = [closing for _, closing in pairs]
        if token in openings:
            stack.append(openings.index(token))
        elif not stack or stack.pop() != closings.index(token):
            return v.INVALID
    return v.INCOMPLETE if stack else v.BALANCED


def point_semantics(node, word, position):
    if node.op == "sym":
        return word[position] == node.value
    if node.op == "const":
        return node.value
    if node.op == "count":
        return sum(point_semantics(node.args[0], word, earlier) for earlier in range(position + 1))
    left = point_semantics(node.args[0], word, position)
    if node.op == "neg":
        return -left
    if node.op == "not":
        return not left
    right = point_semantics(node.args[1], word, position)
    if node.op == "add":
        return left + right
    if node.op == "lt":
        return left < right
    return left and right


class PrefixProgramTests(unittest.TestCase):
    def test_dyck_two_types_exhaustive_short_prefixes_against_independent_stack(self):
        pairs = v.bracket_pairs(2)
        alphabet = (*pairs[0], *pairs[1], v.NEUTRAL)
        for size in range(1, 6):
            for word in itertools.product(alphabet, repeat=size):
                expected = [v.IGNORE] + [typed_stack_status(word[:end], pairs) for end in range(1, size + 1)]
                self.assertEqual(typed_targets((v.BOS, *word), 2), expected)

    def test_crossed_brackets_are_invalid_despite_each_type_balancing(self):
        (a, close_a), (b, close_b) = v.bracket_pairs(2)
        self.assertEqual(typed_targets((v.BOS, a, b, close_a, close_b), 2),
                         [v.IGNORE, v.INCOMPLETE, v.INCOMPLETE, v.INVALID, v.INVALID])

    def test_typed_invalid_prefix_never_recovers_after_closes_or_neutrals(self):
        (a, close_a), (_, close_b) = v.bracket_pairs(2)
        self.assertEqual(typed_targets((v.BOS, a, close_b, v.NEUTRAL, close_a), 2)[2:],
                         [v.INVALID, v.INVALID, v.INVALID])

    def test_neutral_insertions_leave_typed_stack_state_unchanged(self):
        (a, close_a), (b, close_b) = v.bracket_pairs(2)
        word = (a, b, close_b, close_a)
        expected = typed_targets((v.BOS, *word), 2)[1:]
        expanded = (v.BOS, *(token for bracket in word for token in (bracket, v.NEUTRAL)))
        result = typed_targets(expanded, 2)[1:]
        self.assertEqual(result[::2], expected)
        self.assertEqual(result[1::2], expected)

    def test_typed_pairs_preserve_counts_and_neutral_locations_but_change_validity(self):
        for types, balance in itertools.product((2, 3, 8), (1, 2, 4)):
            spec = TaskSpec(task="dyck2", length=32, bracket_types=types, max_balance=balance)
            for seed in range(12):
                first, second = typed_pair(spec, random.Random(seed), 32)
                self.assertEqual(Counter(first.tokens), Counter(second.tokens))
                self.assertEqual([i for i, x in enumerate(first.tokens) if x == v.NEUTRAL],
                                 [i for i, x in enumerate(second.tokens) if x == v.NEUTRAL])
                self.assertEqual(first.targets[-1], v.BALANCED)
                self.assertEqual(second.targets[-1], v.INVALID)

    def test_typed_generator_respects_balance_gap_and_type_limits(self):
        spec = TaskSpec(task="dyck2", length=40, max_balance=2, bracket_types=3,
                        neutral_fraction=0.4, max_neutral_gap=2)
        pairs = v.bracket_pairs(3)
        openings = {opening for opening, _ in pairs}
        closings = {closing for _, closing in pairs}
        for row in build_split(spec, "train", 7, 64).examples:
            balance = gap = 0
            for token in row.tokens[1:]:
                balance += (token in openings) - (token in closings)
                gap = gap + 1 if token == v.NEUTRAL else 0
                self.assertLessEqual(abs(balance), 2)
                self.assertLessEqual(gap, 2)

    def test_typed_oracle_rejects_unrecognized_bracket_types(self):
        with self.assertRaises(ValueError):
            typed_targets((v.BOS, v.IDENTITY_BASE + 5), 2)

    def test_formula_primitives_match_recursive_point_semantics(self):
        a, b, c = (Node("sym", value=token) for token in (v.A, v.B, v.C))
        count_a, count_b = Node("count", (a,)), Node("count", (b,))
        difference = Node("add", (count_a, Node("neg", (count_b,))))
        comparison = Node("lt", (difference, Node("const", value=-1)))
        nested = Node("lt", (Node("count", (Node("not", (comparison,)),)),
                             Node("add", (Node("count", (c,)), Node("const", value=2)))))
        formulas = (a, comparison, Node("not", (comparison,)), nested, Node("and", (comparison, c)))
        for word in itertools.product((v.A, v.B, v.C), repeat=4):
            for formula in formulas:
                expected = tuple(point_semantics(formula, word, position) for position in range(4))
                self.assertEqual(evaluate_formula(formula, word), expected)

    def test_formula_counts_include_the_current_position(self):
        atom = Node("sym", value=v.A)
        comparison = Node("lt", (Node("const", value=0), Node("count", (atom,))))
        self.assertEqual(evaluate_formula(comparison, (v.B, v.A, v.C)), (False, True, True))

    def test_count_depth_is_structural_and_ignores_boolean_combination_depth(self):
        atom = Node("sym", value=v.A)
        counted = Node("count", (atom,))
        condition = Node("lt", (counted, Node("const", value=1)))
        combined = Node("and", (Node("not", (condition,)), condition))
        nested = Node("lt", (Node("count", (combined,)), counted))
        self.assertEqual(atom.depth, 0)
        self.assertEqual(combined.depth, 1)
        self.assertEqual(nested.depth, 2)

    def test_generated_programs_have_requested_depth_and_both_truth_values(self):
        for depth in range(1, 6):
            for seed in range(8):
                formula, witnesses = program_with_witnesses(depth, seed)
                self.assertEqual(formula.depth, depth)
                self.assertFalse(evaluate_formula(formula, witnesses[0])[-1])
                self.assertTrue(evaluate_formula(formula, witnesses[1])[-1])
                self.assertEqual(program_with_witnesses(depth, seed), (formula, witnesses))
                self.assertIn("op", asdict(formula))
        descriptions = {program_with_witnesses(2, seed)[0].describe() for seed in range(12)}
        self.assertGreater(len(descriptions), 8)

    def test_generated_programs_match_point_semantics_and_are_causal(self):
        for depth in (1, 2, 3):
            spec = TaskSpec(task="crasp", length=12, formula_depth=depth, formula_seed=7)
            formula, _ = program_with_witnesses(depth, 7)
            for word in itertools.product((v.A, v.B, v.C), repeat=4):
                prefix = (v.BOS, *word)
                expected = [v.IGNORE] + [v.ACCEPT if point_semantics(formula, word, i) else v.REJECT for i in range(4)]
                self.assertEqual(crasp_targets(prefix, spec), expected)
                self.assertEqual(crasp_targets((*prefix, v.C, v.B, v.A), spec)[:5], expected)

    def test_crasp_labels_and_program_are_fixed_across_data_splits_and_lengths(self):
        spec = TaskSpec(task="crasp", length=24, formula_depth=3, formula_seed=3)
        formula, _ = program_with_witnesses(3, 3)
        for name in ("train", "validation", "test"):
            for row in build_split(spec, name, 10, 12).examples:
                self.assertEqual(row.targets[1:], tuple(v.ACCEPT if truth else v.REJECT
                                                       for truth in evaluate_formula(formula, row.tokens[1:])))
                self.assertTrue({v.ACCEPT, v.REJECT}.isdisjoint(row.tokens))

    def test_saved_formula_ast_can_be_loaded_and_executed(self):
        formula, witnesses = program_with_witnesses(3, 4)
        loaded = Node.from_dict(json.loads(json.dumps(asdict(formula))))
        self.assertEqual(loaded, formula)
        for word in witnesses:
            self.assertEqual(evaluate_formula(loaded, word), evaluate_formula(formula, word))

    def test_formula_invalid_operations_types_and_alphabets_fail(self):
        atom, term = Node("sym", value=v.A), Node("const", value=0)
        for build in (lambda: Node("loop"), lambda: Node("count", (term,)),
                      lambda: Node("add", (atom, term)), lambda: Node("and", (term, atom)),
                      lambda: Node("sym", value=v.NEUTRAL), lambda: Node("lt", (term,))):
            with self.assertRaises(ValueError):
                build()
        with self.assertRaises(ValueError):
            evaluate_formula(atom, (v.ACCEPT,))
        with self.assertRaises(ValueError):
            evaluate_formula(term, (v.A,))
