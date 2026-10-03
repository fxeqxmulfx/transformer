"""Decimal addition, its formats, and carry chains controlled apart from the digits.

A port of `experiments/synthetic_trainers/arithmetic.py`. Source:
arXiv:2310.16028v1, Section 5 and Appendix B. Operands have one extra
leading zero, and the sum keeps that padded width.
"""

from itertools import pairwise

from .base import Answered
from .vocabulary import BOS, DIGIT_BASE, EOS, PLUS, SEP


def carry_profile(left, right):
    """Every outgoing carry, from the least significant digit to the most."""
    carry, result = 0, []
    for a, b in zip(reversed(left), reversed(right)):
        carry = int(a + b + carry >= 10)
        result.append(carry)
    return tuple(result)


def longest_carry(left, right):
    longest = current = 0
    for carry in carry_profile(left, right):
        current = current + 1 if carry else 0
        longest = max(longest, current)
    return longest


class Addition(Answered):
    name = "addition"
    formats = ("addition_order", "index_hints", "symbols", "number_limit")

    def controls(self):
        task = self.task
        numbers = {"symbols": task.symbols, "number_limit": task.number_limit} if task.hints else {}
        return {"addition_order": task.order, "index_hints": task.hints, "carry_sampling": task.carries,
                "carry_length": task.carry_length, **numbers}

    def prompt(self, rng, size, hint_rng):
        task = self.task
        if task.carry_length is None and task.carries == "standard":
            operands = []
            for _ in range(2):
                width = rng.randint(1, size)
                digits = [rng.randint(1, 9)] + [rng.randrange(10) for _ in range(width - 1)]
                operands.append([0] * (size + 1 - width) + digits)
            left, right = operands
        else:
            length = task.carry_length if task.carry_length is not None else rng.randint(0, size)
            start = rng.randint(0, size - length)
            left, right = [], []
            for position in range(size):
                # Outside the prescribed chain, even an incoming carry must stop.
                minimum, maximum = ((10, 18) if length and position == start
                                    else (9, 9) if start < position < start + length
                                    else (0, 8))
                a, b = rng.choice([(a, b) for a in range(10) for b in range(10) if minimum <= a + b <= maximum])
                left.append(a)
                right.append(b)
            left, right = [0, *reversed(left)], [0, *reversed(right)]
        hints = []
        if task.hints:
            start = hint_rng.randint(0, task.number_limit - size)
            hints = [self.number_base + start + i for i in range(size + 1)]
        prompt = [BOS]
        for side, digits in enumerate((left, right)):
            if side:
                prompt.append(PLUS)
            for index, digit in enumerate(digits):
                if hints:
                    prompt.append(hints[index])
                prompt.append(DIGIT_BASE + digit)
        return (*prompt, SEP)

    def inputs(self, prompt):
        """The two padded operands of a prompt, as digits, and their index hints."""
        if not prompt or prompt[0] != BOS or prompt[-1] != SEP or prompt.count(PLUS) != 1:
            raise ValueError("Addition needs BOS, two operands, PLUS, and SEP")
        split = prompt.index(PLUS)
        operands, hints = [], []
        for side in (prompt[1:split], prompt[split + 1:-1]):
            if self.task.hints:
                if len(side) % 2:
                    raise ValueError("Index hints must precede every digit")
                indices, digits = side[::2], side[1::2]
                if not indices or any(not self.number_base <= x <= self.number_base + self.task.number_limit
                                      for x in indices):
                    raise ValueError("Invalid addition index hint")
                if any(b != a + 1 for a, b in pairwise(indices)):
                    raise ValueError("Addition hints must form a contiguous ascending interval")
                hints.append(tuple(indices))
            else:
                digits = side
            if len(digits) < 2 or digits[0] != DIGIT_BASE:
                raise ValueError("Operands need one leading padding zero")
            if any(not DIGIT_BASE <= x < DIGIT_BASE + 10 for x in digits):
                raise ValueError("Invalid decimal digit")
            operands.append(tuple(x - DIGIT_BASE for x in digits))
        if len(operands[0]) != len(operands[1]) or hints and hints[0] != hints[1]:
            raise ValueError("Operands need equal padded widths and corresponding hints")
        return operands[0], operands[1], hints[0] if hints else ()

    def problem_size(self, prompt):
        return len(self.inputs(prompt)[0]) - 1

    def limit(self, prompt):
        return (2 if self.task.hints else 1) * (self.problem_size(prompt) + 1) + 1

    def solve(self, prompt):
        left, right, hints = self.inputs(prompt)
        # Integer arithmetic is an independent oracle for the digit-wise sampler.
        result = str(int("".join(map(str, left))) + int("".join(map(str, right)))).zfill(len(left))
        indices = range(len(result)) if self.task.order == "forward" else range(len(result) - 1, -1, -1)
        answer = []
        for index in indices:
            if hints:
                answer.append(hints[index])
            answer.append(DIGIT_BASE + int(result[index]))
        return (*answer, EOS)

    def check(self, example, expected):
        if self.task.carry_length is not None:
            left, right, _ = self.inputs(example.prompt)
            if longest_carry(left, right) != self.task.carry_length:
                raise ValueError("Addition input violates the requested carry length")

    def content(self, example, target):
        return range(DIGIT_BASE, DIGIT_BASE + 10) if DIGIT_BASE <= target < DIGIT_BASE + 10 else ()
