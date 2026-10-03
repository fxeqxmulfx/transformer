"""Dyck-i prefix status, with i configurable from two through eight.

Source: arXiv:2106.06981v2, Section 5, Dyck-i PTF (i = 1, 2).
More than two types are an experimental extension of that benchmark.
"""

from . import vocabulary as v
from .prefixes import balanced_word
from .records import example_from_targets
from .sampling import insert_gaps, neutral_gaps


def typed_targets(tokens, types):
    if not tokens or tokens[0] != v.BOS:
        raise ValueError("Typed Dyck needs BOS")
    pairs = v.bracket_pairs(types)
    opens = {opening: closing for opening, closing in pairs}
    closes = set(opens.values())
    stack, invalid, targets = [], False, [v.IGNORE]
    for token in tokens[1:]:
        if token == v.NEUTRAL:
            pass
        elif token in opens:
            stack.append(opens[token])
        elif token in closes:
            if not stack or stack.pop() != token:
                invalid = True
        else:
            raise ValueError("Invalid typed bracket token")
        targets.append(v.INVALID if invalid else v.INCOMPLETE if stack else v.BALANCED)
    return targets


def typed_pair(spec, rng, length):
    size = spec.active_size(length)
    pairs = v.bracket_pairs(spec.bracket_types)
    skeleton = balanced_word(size, spec.max_balance, rng)
    word, stack, open_count = [], [], 0
    for token in skeleton:
        if token == v.OPEN:
            kind = open_count if open_count < 2 else rng.randrange(spec.bracket_types)
            open_count += 1
            opening, closing = pairs[kind]
            word.append(opening)
            stack.append(closing)
        else:
            word.append(stack.pop())
    closing_positions = [i for i, token in enumerate(word) if token in {pair[1] for pair in pairs}]
    first = rng.choice(closing_positions)
    second = rng.choice([i for i in closing_positions if word[i] != word[first]])
    corrupted = word.copy()
    corrupted[first], corrupted[second] = corrupted[second], corrupted[first]
    gaps = neutral_gaps(size, length - 1, rng, spec.max_neutral_gap)
    examples = []
    for active in (word, corrupted):
        tokens = [v.BOS, *insert_gaps(active, gaps, v.NEUTRAL)]
        examples.append(example_from_targets(spec.task, tokens, typed_targets(tokens, spec.bracket_types)))
    return tuple(examples)


def random_typed(spec, rng, length):
    pairs = v.bracket_pairs(spec.bracket_types)
    active, balance = [], 0
    for _ in range(spec.active_size(length)):
        opening = balance == -spec.max_balance or (balance < spec.max_balance and rng.random() < 0.5)
        active.append(rng.choice(pairs)[0 if opening else 1])
        balance += 1 if opening else -1
    gaps = neutral_gaps(len(active), length - 1, rng, spec.max_neutral_gap)
    tokens = [v.BOS, *insert_gaps(active, gaps, v.NEUTRAL)]
    return example_from_targets(spec.task, tokens, typed_targets(tokens, spec.bracket_types))
