"""Executable past-counting formulas and a reproducible program generator.

Source: arXiv:2506.16055v3, Section 2.3, syntax of TL[past-count].
Depth is syntactic count nesting; it does not assert a minimal model depth.
One sampled formula stays fixed across every split and evaluation length.
"""

from dataclasses import dataclass
from functools import lru_cache
import itertools
import random

from . import vocabulary as v
from .records import example_from_targets

ARITIES = {"sym": 0, "const": 0, "count": 1, "neg": 1, "add": 2,
           "lt": 2, "not": 1, "and": 2}


@dataclass(frozen=True)
class Node:
    op: str
    args: tuple = ()
    value: int | None = None

    def __post_init__(self):
        if self.op not in ARITIES or len(self.args) != ARITIES[self.op]:
            raise ValueError("Invalid formula operation or arity")
        if not isinstance(self.args, tuple) or any(not isinstance(arg, Node) for arg in self.args):
            raise ValueError("Formula arguments must be an immutable tuple of nodes")
        if self.op in ("sym", "const"):
            if type(self.value) is not int or self.op == "sym" and self.value not in (v.A, v.B, v.C):
                raise ValueError("Invalid atom or integer constant")
        elif self.value is not None:
            raise ValueError("Only atoms and constants carry a value")
        kinds = tuple(arg.kind for arg in self.args)
        expected = {"count": ("formula",), "neg": ("term",), "add": ("term", "term"),
                    "lt": ("term", "term"), "not": ("formula",), "and": ("formula", "formula")}
        if self.op in expected and kinds != expected[self.op]:
            raise ValueError("Formula and term types do not match")

    @property
    def kind(self):
        return "term" if self.op in ("const", "count", "neg", "add") else "formula"

    @property
    def depth(self):
        return max((arg.depth for arg in self.args), default=0) + (self.op == "count")

    def describe(self):
        if self.op == "sym":
            return v.token_name(self.value)
        if self.op == "const":
            return str(self.value)
        return f"{self.op}({', '.join(arg.describe() for arg in self.args)})"

    @classmethod
    def from_dict(cls, payload):
        return cls(payload["op"], tuple(cls.from_dict(arg) for arg in payload["args"]), payload["value"])


def evaluate_formula(formula, word):
    """Evaluate every inclusive prefix using unbounded Python integer counts."""
    if formula.kind != "formula" or any(x not in (v.A, v.B, v.C) for x in word):
        raise ValueError("Need a Boolean formula and its three-symbol alphabet")
    memo = {}

    def visit(node):
        if node in memo:
            return memo[node]
        args = [visit(arg) for arg in node.args]
        if node.op == "sym":
            result = [token == node.value for token in word]
        elif node.op == "const":
            result = [node.value] * len(word)
        elif node.op == "count":
            total, result = 0, []
            for truth in args[0]:
                total += truth
                result.append(total)
        elif node.op == "neg":
            result = [-x for x in args[0]]
        elif node.op == "not":
            result = [not x for x in args[0]]
        elif node.op == "add":
            result = [a + b for a, b in zip(*args)]
        elif node.op == "lt":
            result = [a < b for a, b in zip(*args)]
        else:
            result = [a and b for a, b in zip(*args)]
        memo[node] = result
        return result

    return tuple(visit(formula))


def random_formula(depth, rng):
    if depth == 0:
        atom = Node("sym", value=rng.choice((v.A, v.B, v.C)))
        return Node("not", (atom,)) if rng.random() < 0.25 else atom
    left = Node("count", (random_formula(depth - 1, rng),))
    right = Node("count", (random_formula(depth - 1, rng),))
    if rng.random() < 0.5:
        right = Node("add", (right, Node("count", (Node("sym", value=rng.choice((v.A, v.B, v.C))),))))
    if rng.random() < 0.25:
        left = Node("add", (left, Node("neg", (right,))))
        right = Node("const", value=rng.randint(-2, 2))
    else:
        right = Node("add", (right, Node("const", value=rng.randint(-2, 2))))
    comparison = Node("lt", (left, right))
    if rng.random() < 0.3:
        comparison = Node("not", (comparison,))
    if rng.random() < 0.3:
        comparison = Node("and", (comparison, random_formula(0, rng)))
    return comparison


@lru_cache(maxsize=128)
def program_with_witnesses(depth, seed):
    if not 1 <= depth <= 5 or seed < 0:
        raise ValueError("Need a count depth in [1, 5] and a nonnegative program seed")
    rng = random.Random(seed)
    words = [word for size in range(1, 5) for word in itertools.product((v.A, v.B, v.C), repeat=size)]
    words += [tuple(rng.choice((v.A, v.B, v.C)) for _ in range(rng.randint(5, 12))) for _ in range(32)]
    # Bounded selection excludes programs constant on the witness bank.
    for _ in range(64):
        formula = random_formula(depth, rng)
        witnesses = {}
        for word in words:
            witnesses.setdefault(evaluate_formula(formula, word)[-1], word)
            if len(witnesses) == 2:
                return formula, (witnesses[False], witnesses[True])
    raise ValueError("Could not sample a nonconstant formula; choose another program seed")


def crasp_targets(tokens, spec):
    if not tokens or tokens[0] != v.BOS:
        raise ValueError("C-RASP prefix input needs BOS")
    formula, _ = program_with_witnesses(spec.formula_depth, spec.formula_seed)
    return [v.IGNORE, *(v.ACCEPT if truth else v.REJECT for truth in evaluate_formula(formula, tokens[1:]))]


def crasp_example(spec, rng, length):
    _, witnesses = program_with_witnesses(spec.formula_depth, spec.formula_seed)
    prefix = list(rng.choice(witnesses)) if rng.random() < 0.5 else []
    word = prefix[:length - 1]
    word.extend(rng.choice((v.A, v.B, v.C)) for _ in range(length - 1 - len(word)))
    tokens = [v.BOS, *word]
    return example_from_targets(spec.task, tokens, crasp_targets(tokens, spec))
