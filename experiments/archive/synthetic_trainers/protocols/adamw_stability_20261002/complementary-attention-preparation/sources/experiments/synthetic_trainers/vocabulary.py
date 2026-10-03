"""Disjoint structural tokens, decimal digits, symbols, and atomic integers."""

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


def token_name(token, spec=None):
    if spec is not None and spec.uses_numbers and spec.number_base <= token <= spec.number_base + spec.number_limit:
        return f"number_{token - spec.number_base}"
    return NAMES.get(token, f"id_{token - IDENTITY_BASE}")


def bracket_pairs(types):
    return ((OPEN, CLOSE), *(tuple(IDENTITY_BASE + 2 * i + j for j in range(2))
                             for i in range(types - 1)))
