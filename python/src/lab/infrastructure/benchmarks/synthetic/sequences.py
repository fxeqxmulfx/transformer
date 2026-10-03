"""RASP and RASP-L tasks over a word of symbols, and counting.

A port of the symbol tasks of `experiments/synthetic_trainers/sequences.py`
and `sequence_oracles.py`. Sources: RASP, arXiv:2106.06981v2, Section 5;
RASP-L, arXiv:2310.16028v1, Sections 4 and 5 and Appendix B.
"""

from collections import Counter

from .base import Answered
from .vocabulary import BOS, EOS, IDENTITY_BASE, SEP


class Symbols(Answered):
    """A word over a pool of at most `pool` symbols, itself drawn from the alphabet."""
    pool = 8

    def prompt(self, rng, size, hint_rng):
        return (*([BOS] if self.bos else []), *self.word(rng, size), SEP)

    def word(self, rng, size):
        alphabet = range(IDENTITY_BASE, self.number_base)
        pool = rng.sample(alphabet, rng.randint(2, min(self.task.symbols, self.pool, max(2, size))))
        return [rng.choice(pool) for _ in range(size)]

    def problem_size(self, prompt):
        body = self.body(prompt)
        if any(not IDENTITY_BASE <= x < self.number_base for x in body):
            raise ValueError("Invalid symbol in sequence input")
        return len(body)

    def limit(self, prompt):
        return self.problem_size(prompt) + 1 + int(self.bos)

    def solve(self, prompt):
        return (*self.write(self.body(prompt)), EOS)

    def content(self, example, target):
        # A number in an answer is a count, which a problem of its length bounds.
        if target >= self.number_base:
            return range(self.number_base + 1, self.number_base + self.problem_size(example.prompt) + 1)
        return super().content(example, target)


class Histogram(Symbols):
    name = "histogram"
    formats = ("histogram_bos", "number_limit")

    @property
    def bos(self):
        return self.task.bos

    def controls(self):
        return {"symbols": self.task.symbols, "histogram_bos": self.task.bos, "number_limit": self.task.number_limit}

    def write(self, word):
        counts = Counter(word)
        return [BOS] * self.bos + [self.number_base + counts[token] for token in word]


class DoubleHistogram(Symbols):
    name = "histogram2"
    formats = ("number_limit",)

    def controls(self):
        return {"symbols": self.task.symbols, "number_limit": self.task.number_limit}

    def write(self, word):
        counts = Counter(word)
        classes = Counter(counts.values())
        return [BOS, *(self.number_base + classes[counts[token]] for token in word)]


class Mode(Symbols):
    name = "mode"
    formats = ("scratchpad", "number_limit")
    pool = 5

    def controls(self):
        numbers = {"number_limit": self.task.number_limit} if self.task.uses_numbers else {}
        return {"symbols": self.task.symbols, "scratchpad": self.task.scratchpad, **numbers}

    def word(self, rng, size):
        word = super().word(rng, size)
        counts = Counter(word)
        highest = max(counts.values())
        winners = [x for x in counts if counts[x] == highest]
        if len(winners) > 1:
            # Move one occurrence to the selected winner; its count is now uniquely highest.
            winner = rng.choice(winners)
            donor = rng.choice([x for x in winners if x != winner])
            positions = [i for i, x in enumerate(word) if x == donor]
            word[rng.choice(positions)] = winner
        return word

    def limit(self, prompt):
        size = self.problem_size(prompt)
        return 2 + (2 * min(size, self.task.symbols) if self.task.uses_numbers else 0)

    def write(self, word):
        counts = Counter(word)
        highest = max(counts.values())
        winners = [token for token, count in counts.items() if count == highest]
        if len(winners) != 1:
            raise ValueError("Mode requires a unique most frequent symbol")
        answer = []
        if self.task.scratchpad != "none":
            order = (sorted(counts, key=lambda x: (counts[x], word.index(x))) if self.task.scratchpad == "counts"
                     else counts)
            for token in order:
                answer.extend((self.number_base + counts[token], token))
        return [*answer, winners[0]]


class MostFrequent(Symbols):
    name = "most_freq"

    def controls(self):
        return {"symbols": self.task.symbols}

    def write(self, word):
        counts = Counter(word)
        order = sorted(counts, key=lambda x: (-counts[x], word.index(x)))
        return [BOS, *order, *([BOS] * (len(word) - len(order)))]


class Rewrite(Symbols):
    """A rewriting of the word, drawn with `unique` from the whole alphabet without replacement."""

    def controls(self):
        return {"symbols": self.task.symbols, "unique": self.task.unique}

    def word(self, rng, size):
        if self.task.unique:
            return rng.sample(range(IDENTITY_BASE, self.number_base), size)
        return super().word(rng, size)

    def problem_size(self, prompt):
        size = super().problem_size(prompt)
        if self.task.unique and len(set(self.body(prompt))) != size:
            raise ValueError("A unique-token problem contains a repeated symbol")
        return size

    def limit(self, prompt):
        return self.problem_size(prompt) + 1


class Copy(Rewrite):
    name = "copy"
    pool = 2

    def write(self, word):
        return word


class Reverse(Rewrite):
    name = "reverse"
    pool = 2

    def write(self, word):
        return word[::-1]


class Sort(Rewrite):
    name = "sort"

    def write(self, word):
        return sorted(word)


class Count(Answered):
    name = "count"

    def controls(self):
        return {"symbols": self.task.symbols, "number_limit": self.task.number_limit}

    def prompt(self, rng, size, hint_rng):
        start = rng.randint(1, self.task.number_limit - size + 1)
        return (BOS, self.number_base + start, self.number_base + start + size - 1, SEP)

    def interval(self, prompt):
        body = self.body(prompt)
        if len(body) != 2 or any(not self.number_base + 1 <= x <= self.number_base + self.task.number_limit
                                 for x in body) or body[1] < body[0]:
            raise ValueError("Count needs a nonempty ascending interval of atomic positive integers")
        return body

    def problem_size(self, prompt):
        start, end = self.interval(prompt)
        return end - start + 1

    def limit(self, prompt):
        return self.problem_size(prompt) + 1

    def solve(self, prompt):
        start, end = self.interval(prompt)
        return (*range(start, end + 1), EOS)

    def content(self, example, target):
        if target >= self.number_base:
            return range(self.number_base + 1, self.number_base + self.task.number_limit + 1)
        return super().content(example, target)
