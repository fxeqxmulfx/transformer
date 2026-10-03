"""Disjoint structural tokens, decimal digits, symbols, and atomic numbers.

The token layout of the historical synthetic suite
(`experiments/synthetic_trainers/vocabulary.py`): structure and labels below
`DIGIT_BASE`, ten digits, then symbols from `IDENTITY_BASE`; a task that
writes numbers places them after its symbols.
"""

PAD, BOS, KEY, VALUE, QUERY, END_TABLE, FILL = range(7)
OPEN, CLOSE, A, B, NEUTRAL = range(7, 12)
BALANCED, INCOMPLETE, INVALID, REJECT, ACCEPT = range(12, 17)
EOS, SEP, COMMA, PLUS, ZERO, ONE, C, EVEN, ODD = range(17, 26)
DIGIT_BASE = 26
IDENTITY_BASE = 36
IGNORE = -100

NAMES = {
    PAD: "pad", BOS: "bos", KEY: "key", VALUE: "value", QUERY: "query",
    END_TABLE: "end_table", FILL: "fill", OPEN: "(", CLOSE: ")",
    A: "a", B: "b", NEUTRAL: "_", BALANCED: "balanced",
    INCOMPLETE: "incomplete", INVALID: "invalid", REJECT: "reject",
    ACCEPT: "accept", EOS: "eos", SEP: "answer", COMMA: ",", PLUS: "+",
    ZERO: "0-bit", ONE: "1-bit", C: "c", EVEN: "even", ODD: "odd",
}
NAMES.update({DIGIT_BASE + digit: str(digit) for digit in range(10)})


def token_name(token, numbers=None):
    """A token's name; `numbers`, the range of a task's number tokens from zero, are named by value."""
    if numbers is not None and token in numbers:
        return f"number_{token - numbers.start}"
    return NAMES.get(token, f"id_{token - IDENTITY_BASE}")


def bracket_pairs(types):
    """Opening and closing tokens of each bracket type: parentheses, then pairs of symbols."""
    return ((OPEN, CLOSE), *(tuple(IDENTITY_BASE + 2 * i + j for j in range(2)) for i in range(types - 1)))
