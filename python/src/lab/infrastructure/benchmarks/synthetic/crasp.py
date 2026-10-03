"""Executable past-counting formulas, and one drawn reproducibly from a seed.

A port of `experiments/synthetic_trainers/crasp.py`. Source: arXiv:2506.16055v3,
Section 2.2, the syntax of TL[past-count]; that module cites Section 2.3,
which is on Parikh vectors. Depth is syntactic count nesting, not the least
depth of an equivalent formula. One formula serves every split and length.
"""

from dataclasses import dataclass
from functools import lru_cache
import itertools
import random

from .base import Generator
from .vocabulary import A, ACCEPT, B, BOS, C, IGNORE, REJECT, token_name

ARITIES = {"sym": 0, "const": 0, "count": 1, "neg": 1, "add": 2, "lt": 2, "not": 1, "and": 2}
SIGNATURES = {"count": ("formula",), "neg": ("term",), "add": ("term", "term"), "lt": ("term", "term"),
              "not": ("formula",), "and": ("formula", "formula")}


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
            if type(self.value) is not int or self.op == "sym" and self.value not in (A, B, C):
                raise ValueError("Invalid atom or integer constant")
        elif self.value is not None:
            raise ValueError("Only atoms and constants carry a value")
        if self.op in SIGNATURES and tuple(arg.kind for arg in self.args) != SIGNATURES[self.op]:
            raise ValueError("Formula and term types do not match")

    @property
    def kind(self):
        return "term" if self.op in ("const", "count", "neg", "add") else "formula"

    @property
    def depth(self):
        return max((arg.depth for arg in self.args), default=0) + (self.op == "count")

    def describe(self):
        if self.op == "sym":
            return token_name(self.value)
        if self.op == "const":
            return str(self.value)
        return f"{self.op}({', '.join(arg.describe() for arg in self.args)})"


def evaluate_formula(formula, word):
    """Evaluate every inclusive prefix using unbounded Python integer counts."""
    if formula.kind != "formula" or any(x not in (A, B, C) for x in word):
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
        atom = Node("sym", value=rng.choice((A, B, C)))
        return Node("not", (atom,)) if rng.random() < 0.25 else atom
    left = Node("count", (random_formula(depth - 1, rng),))
    right = Node("count", (random_formula(depth - 1, rng),))
    if rng.random() < 0.5:
        right = Node("add", (right, Node("count", (Node("sym", value=rng.choice((A, B, C))),))))
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
    """The formula of `depth` drawn from `seed`, and a word it rejects and one it accepts."""
    if not 1 <= depth <= 5 or seed < 0:
        raise ValueError("Need a count depth in [1, 5] and a nonnegative program seed")
    rng = random.Random(seed)
    words = [word for size in range(1, 5) for word in itertools.product((A, B, C), repeat=size)]
    words += [tuple(rng.choice((A, B, C)) for _ in range(rng.randint(5, 12))) for _ in range(32)]
    # Bounded selection excludes programs constant on the witness bank.
    for _ in range(64):
        formula = random_formula(depth, rng)
        witnesses = {}
        for word in words:
            witnesses.setdefault(evaluate_formula(formula, word)[-1], word)
            if len(witnesses) == 2:
                return formula, (witnesses[False], witnesses[True])
    raise ValueError("Could not sample a nonconstant formula; choose another program seed")


class CRASP(Generator):
    name = "crasp"

    def controls(self):
        return {"formula_depth": self.task.depth, "formula_seed": self.task.program}

    @property
    def program(self):
        return program_with_witnesses(self.task.depth, self.task.program)

    def pair(self, rng, length):
        return self.sample(rng, length), self.sample(rng, length)

    def sample(self, rng, length):
        _, witnesses = self.program
        prefix = list(rng.choice(witnesses)) if rng.random() < 0.5 else []
        word = prefix[:length - 1]
        word.extend(rng.choice((A, B, C)) for _ in range(length - 1 - len(word)))
        return self.example([BOS, *word])

    def targets(self, tokens):
        if not tokens or tokens[0] != BOS:
            raise ValueError("C-RASP prefix input needs BOS")
        formula, _ = self.program
        return [IGNORE, *(ACCEPT if truth else REJECT for truth in evaluate_formula(formula, tokens[1:]))]

    def content(self, example, target):
        return (REJECT, ACCEPT)
