"""Generators: a task at given problem lengths, sampled, and checked by an oracle on the raw input.

A port of the historical synthetic suite (`experiments/synthetic_trainers`:
`data.py`, `oracles.py`, `sequences.py`, `sequence_oracles.py`,
`label_noise.py`, `corpus.py`). Every draw from the random generator is the
historical one, in the historical order, so a split is the historical split
and has its fingerprint.
"""

import random

from .records import example_from_targets, generation_example
from .vocabulary import BOS, COMMA, EOS, IDENTITY_BASE, IGNORE, SEP, token_name


class Generator:
    """A task at problem lengths `min_length` to `length` that labels its own input.

    `name` and `controls` are the task's historical name and the controls that
    seed and fingerprint its splits; `formats` are the controls that write a
    problem without changing it. A split's seed leaves them out, so the
    formats of one problem are trained and tested on the same problems.
    """
    name = None
    formats = ()
    vocab = IDENTITY_BASE
    numbers = None

    def __init__(self, problem):
        self.task, self.length, self.min_length = problem
        self.minimum = problem.minimum
        self.context = self.task.context(self.length)

    def controls(self):
        return {}

    def identity(self):
        """What a split's fingerprint records of its distribution."""
        return {"task": self.name, "length": self.length, "min_length": self.min_length, **self.controls()}

    def problem_identity(self):
        """What seeds a split: its distribution, without the formats."""
        return {key: value for key, value in self.identity().items() if key not in self.formats}

    def draw(self, rng):
        """The next rows of a split: two problems of one length, in random order, so row parity hides labels."""
        length = rng.randint(self.minimum, self.length)
        pair = self.pair(rng, length)
        return pair[::-1] if rng.random() < 0.5 else pair

    def example(self, tokens):
        return example_from_targets(self.name, tokens, self.targets(tokens))

    def validate(self, example):
        """Reject an example outside the distribution, or whose targets the raw-input oracle disputes."""
        size = self.size(example)
        if example.task != self.name or not self.minimum <= size <= self.length:
            raise ValueError("Example task or length does not match the specification")
        if min(example.tokens) < 0 or max(example.tokens) >= self.vocab:
            raise ValueError("Input token is outside the vocabulary")
        expected = self.targets(example.tokens)
        if tuple(expected) != example.targets:
            raise ValueError("Targets disagree with the raw-input oracle")
        if len(example.tokens) > self.context:
            raise ValueError("Serialized input exceeds the context bound")
        self.check(example, expected)
        previous = None
        for target, change in zip(expected, example.changes):
            if change != (target != IGNORE and (previous is None or previous != target)):
                raise ValueError("Incorrect state-change mask")
            if target != IGNORE:
                previous = target

    def size(self, example):
        """The problem length of a valid example."""
        if example.prompt:
            raise ValueError("A prefix task cannot contain a generated answer")
        return len(example.tokens)

    def check(self, example, expected):
        """Constraints of the task's own on a valid example."""

    def key(self, example):
        """The input that identifies a problem: its prompt, or its whole input."""
        return example.prompt or example.tokens

    def labels(self, example, position):
        """The legal values of a target, which label noise draws from; none for a structural target."""
        target = example.targets[position]
        if target in (IGNORE, BOS, EOS, COMMA, SEP):
            return ()
        return self.content(example, target)

    def token_name(self, token):
        return token_name(token, self.numbers)


class Answered(Generator):
    """A task whose answer follows its prompt and a separator, ends with EOS, and follows from the prompt alone.

    Every task reserves `symbols` symbol tokens; a task that writes numbers
    places `number_limit + 1` number tokens, from zero, after them.
    """
    bos = True

    @property
    def number_base(self):
        return IDENTITY_BASE + self.task.symbols

    @property
    def vocab(self):
        return self.number_base + self.task.number_limit + 1 if self.task.uses_numbers else self.number_base

    @property
    def numbers(self):
        return range(self.number_base, self.vocab) if self.task.uses_numbers else None

    def pair(self, rng, length):
        return self.sample(rng, length), self.sample(rng, length)

    def sample(self, rng, size):
        # Always consume one independent format seed, even without positional hints.
        hint_rng = random.Random(rng.getrandbits(128))
        prompt = self.prompt(rng, size, hint_rng)
        return generation_example(self.name, prompt, self.answer(prompt), self.limit(prompt))

    def answer(self, prompt):
        """The answer to a raw prompt, which is checked first."""
        self.problem_size(prompt)
        return self.solve(prompt)

    def targets(self, tokens):
        # Teacher-forced tokens are checked by locating the unique separator.
        if SEP not in tokens:
            raise ValueError("Missing answer separator")
        end = tokens.index(SEP) + 1
        prompt = tokens[:end]
        answer = self.answer(prompt)
        if tuple(tokens) != (*prompt, *answer[:-1]):
            raise ValueError("Teacher-forcing input disagrees with the raw-prompt answer")
        return [IGNORE] * (end - 1) + list(answer)

    def size(self, example):
        size = self.problem_size(example.prompt)
        if example.answer != self.answer(example.prompt):
            raise ValueError("Generated answer disagrees with the raw-prompt oracle")
        if example.generation_limit != self.limit(example.prompt):
            raise ValueError("Generation limit must depend only on the prompt format")
        return size

    def body(self, prompt):
        """The problem between BOS, where the task has one, and the final separator."""
        if not prompt or prompt[-1] != SEP or prompt.count(SEP) != 1:
            raise ValueError("A prompt needs exactly one final answer separator")
        if self.bos and prompt[0] != BOS:
            raise ValueError("Missing BOS")
        body = tuple(prompt[int(self.bos):-1])
        if not body:
            raise ValueError("An empty problem is unsupported")
        return body

    def content(self, example, target):
        return range(IDENTITY_BASE, self.number_base)
