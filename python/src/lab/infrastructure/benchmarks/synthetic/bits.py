"""RASP-L tasks over a word of bits: parity, with its scratchpads, and Boolean AND.

A port of the bit tasks of `experiments/synthetic_trainers/sequences.py` and
`sequence_oracles.py`. Source: arXiv:2310.16028v1, Section 5 and Appendix B.
"""

from itertools import pairwise

from .base import Answered
from .vocabulary import ACCEPT, BOS, EOS, EVEN, ODD, ONE, REJECT, SEP, ZERO


class Bits(Answered):
    """A word of bits, each preceded with `hinted` by its index as a number token."""
    hinted = False

    def prompt(self, rng, size, hint_rng):
        bits = self.bits(rng, size)
        prompt = [BOS]
        start = hint_rng.randint(0, self.task.number_limit - size) if self.hinted else 0
        for index, bit in enumerate(bits):
            if self.hinted:
                prompt.append(self.number_base + start + index)
            prompt.append(bit)
        return (*prompt, SEP)

    def inputs(self, prompt):
        """The bits of a prompt, and their index hints."""
        body = self.body(prompt)
        if self.hinted:
            if len(body) % 2:
                raise ValueError("Index hints must precede each bit")
            hints, bits = body[::2], body[1::2]
            if any(not self.number_base <= x <= self.number_base + self.task.number_limit for x in hints):
                raise ValueError("Invalid bit index hint")
            if any(b != a + 1 for a, b in pairwise(hints)):
                raise ValueError("Bit hints must form a contiguous ascending interval")
        else:
            hints, bits = (), body
        if not {ZERO, ONE}.issuperset(bits):
            raise ValueError("Invalid bit input")
        return bits, hints

    def problem_size(self, prompt):
        return len(self.inputs(prompt)[0])


class Parity(Bits):
    name = "parity"
    formats = ("scratchpad", "index_hints", "symbols", "number_limit")

    @property
    def hinted(self):
        return self.task.hints

    def controls(self):
        task = self.task
        numbers = {"symbols": task.symbols, "number_limit": task.number_limit} if task.hints else {}
        return {"scratchpad": task.scratchpad, "index_hints": task.hints, **numbers}

    def bits(self, rng, size):
        return [rng.choice((ZERO, ONE)) for _ in range(size)]

    def limit(self, prompt):
        if self.task.scratchpad == "none":
            return 2
        return (2 if self.task.hints else 1) * self.problem_size(prompt) + 2

    def solve(self, prompt):
        bits, hints = self.inputs(prompt)
        scratchpad = self.task.scratchpad
        parity = 0
        answer = [EVEN] if scratchpad != "none" else []
        for index, bit in enumerate(bits):
            parity ^= bit == ONE
            if scratchpad == "running" or scratchpad == "ones" and bit == ONE:
                if hints:
                    answer.append(hints[index])
                answer.append(ODD if parity else EVEN)
        if scratchpad == "none":
            answer.append(ODD if parity else EVEN)
        return (*answer, EOS)

    def content(self, example, target):
        return (EVEN, ODD) if target in (EVEN, ODD) else ()


class BooleanAnd(Bits):
    name = "boolean_and"

    def controls(self):
        return {"and_shift": self.task.shift, **({"and_region": self.task.region} if self.task.shift else {})}

    def bits(self, rng, size):
        bits = [ONE] * size
        if rng.random() < 0.5:
            bits[rng.choice(self.zeros(size))] = ZERO
        return bits

    def zeros(self, size):
        """Where the zero may be: anywhere, or with `shift` before or in the last quarter of the bits."""
        if not self.task.shift:
            return range(size)
        cut = size - max(1, size // 4)
        return range(cut) if self.task.region == "early" else range(cut, size)

    def limit(self, prompt):
        return 2

    def solve(self, prompt):
        bits, _ = self.inputs(prompt)
        return (ACCEPT if all(x == ONE for x in bits) else REJECT, EOS)

    def check(self, example, expected):
        bits, _ = self.inputs(example.prompt)
        zeros = [i for i, token in enumerate(bits) if token == ZERO]
        if len(zeros) > 1 or any(i not in self.zeros(len(bits)) for i in zeros):
            raise ValueError("AND input violates its zero-position distribution")

    def content(self, example, target):
        return (REJECT, ACCEPT)
