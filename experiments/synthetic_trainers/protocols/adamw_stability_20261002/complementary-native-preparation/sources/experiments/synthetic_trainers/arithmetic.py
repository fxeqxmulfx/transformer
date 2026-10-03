"""Decimal addition formats and independently controlled carry chains.

Source: arXiv:2310.16028v1, Section 5 and Appendix experimental details.
Operands have one extra leading zero; answers retain that padded width.
"""

from . import vocabulary as v


def addition_inputs(prompt, spec):
    if not prompt or prompt[0] != v.BOS or prompt[-1] != v.SEP or prompt.count(v.PLUS) != 1:
        raise ValueError("Addition needs BOS, two operands, PLUS, and SEP")
    split = prompt.index(v.PLUS)
    operands, hints = [], []
    for side in (prompt[1:split], prompt[split + 1:-1]):
        if spec.index_hints:
            if len(side) % 2:
                raise ValueError("Index hints must precede every digit")
            indices, digits = side[::2], side[1::2]
            if not indices or any(not spec.number_base <= x <= spec.number_base + spec.number_limit
                                  for x in indices):
                raise ValueError("Invalid addition index hint")
            if any(b != a + 1 for a, b in zip(indices, indices[1:])):
                raise ValueError("Addition hints must form a contiguous ascending interval")
            hints.append(tuple(indices))
        else:
            digits = side
        if len(digits) < 2 or digits[0] != v.DIGIT_BASE:
            raise ValueError("Operands need one leading padding zero")
        if any(not v.DIGIT_BASE <= x < v.DIGIT_BASE + 10 for x in digits):
            raise ValueError("Invalid decimal digit")
        operands.append(tuple(x - v.DIGIT_BASE for x in digits))
    if len(operands[0]) != len(operands[1]) or hints and hints[0] != hints[1]:
        raise ValueError("Operands need equal padded widths and corresponding hints")
    return operands[0], operands[1], hints[0] if hints else ()


def carry_profile(left, right):
    """Return every outgoing carry, ordered from least significant to most."""
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


def addition_answer(prompt, spec):
    left, right, hints = addition_inputs(prompt, spec)
    # Integer arithmetic is an independent oracle for the digit-wise sampler.
    a = int("".join(map(str, left)))
    b = int("".join(map(str, right)))
    result = str(a + b).zfill(len(left))
    indices = range(len(result)) if spec.addition_order == "forward" else range(len(result) - 1, -1, -1)
    answer = []
    for index in indices:
        if hints:
            answer.append(hints[index])
        answer.append(v.DIGIT_BASE + int(result[index]))
    return (*answer, v.EOS)


def addition_prompt(spec, rng, size, hint_rng=None):
    if spec.carry_length is None and spec.carry_sampling == "standard":
        operands = []
        for _ in range(2):
            width = rng.randint(1, size)
            digits = [rng.randint(1, 9)] + [rng.randrange(10) for _ in range(width - 1)]
            operands.append([0] * (size + 1 - width) + digits)
        left, right = operands
    else:
        length = spec.carry_length if spec.carry_length is not None else rng.randint(0, size)
        start = rng.randint(0, size - length)
        left, right = [], []
        for position in range(size):
            # Outside the prescribed chain, even an incoming carry must stop.
            minimum, maximum = ((10, 18) if length and position == start
                                else (9, 9) if start < position < start + length
                                else (0, 8))
            pairs = [(a, b) for a in range(10) for b in range(10) if minimum <= a + b <= maximum]
            a, b = rng.choice(pairs)
            left.append(a)
            right.append(b)
        left, right = [0, *reversed(left)], [0, *reversed(right)]
    hints = []
    if spec.index_hints:
        start = (hint_rng or rng).randint(0, spec.number_limit - size)
        hints = [spec.number_base + start + i for i in range(size + 1)]
    prompt = [v.BOS]
    for side_index, digits in enumerate((left, right)):
        if side_index:
            prompt.append(v.PLUS)
        for index, digit in enumerate(digits):
            if hints:
                prompt.append(hints[index])
            prompt.append(v.DIGIT_BASE + digit)
    return (*prompt, v.SEP)
