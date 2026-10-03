"""Matched prefix examples: preserve frequencies while changing the answer."""

from . import vocabulary as v
from .oracles import oracle_targets
from .records import example_from_targets
from .sampling import distribute, insert_gaps, neutral_gaps


def balanced_word(size, maximum, rng):
    if size < 0 or size % 2 or maximum < 1:
        raise ValueError("A balanced word needs an even size and positive balance bound")
    word, balance = [], 0
    for index in range(size):
        remaining = size - index
        opening = balance == 0 or (balance < maximum and balance < remaining and rng.random() < 0.5)
        word.append(v.OPEN if opening else v.CLOSE)
        balance += 1 if opening else -1
    return word


def dyck_pair(spec, rng, size):
    # The first component varies the location of the first violation.
    cut = 2 * rng.randint(1, (size - 4) // 2)
    first = balanced_word(cut, spec.max_balance, rng) + balanced_word(size - cut, spec.max_balance, rng)
    second = first.copy()
    candidates = [i for i in range(cut + 1, size - 1) if first[i] == v.CLOSE]
    closing = candidates[0] if spec.max_balance == 1 else rng.choice(candidates)
    second[cut], second[closing] = second[closing], second[cut]
    return first, second


def blocks_pair(spec, rng, size):
    if spec.blocks == 1:
        first = [v.A] * size
        second = first.copy()
        second[rng.randrange(size)] = v.B
        return first, second
    lengths = [2, 2] + [1] * (spec.blocks - 2)
    extra = distribute(size - sum(lengths), spec.blocks, rng)
    first = []
    for index, (base, added) in enumerate(zip(lengths, extra)):
        first.extend([v.A if index % 2 == 0 else v.B] * (base + added))
    # Splitting the first boundary introduces two runs with identical counts.
    second = first.copy()
    boundary = lengths[0] + extra[0]
    second[boundary - 1], second[boundary] = second[boundary], second[boundary - 1]
    return first, second


def prefix_pair(spec, rng, length):
    size = spec.active_size(length)
    active = dyck_pair(spec, rng, size) if spec.task == "dyck" else blocks_pair(spec, rng, size)
    gaps = neutral_gaps(size, length - 1, rng, spec.max_neutral_gap)
    examples = []
    for word in active:
        tokens = [v.BOS] + insert_gaps(word, gaps, v.NEUTRAL)
        examples.append(example_from_targets(spec.task, tokens, oracle_targets(tokens, spec)))
    return tuple(examples)


def random_prefix(spec, rng, length):
    """Broaden support beyond matched pairs and their specific corruptions."""
    size = spec.active_size(length)
    if spec.task == "dyck":
        active, balance = [], 0
        for _ in range(size):
            opening = balance == -spec.max_balance or (balance < spec.max_balance and rng.random() < 0.5)
            active.append(v.OPEN if opening else v.CLOSE)
            balance += 1 if opening else -1
    else:
        runs = rng.randint(1, min(size, 2 * spec.blocks + 3))
        lengths = distribute(size - runs, runs, rng)
        start = rng.choice((v.A, v.B))
        active = []
        for index, added in enumerate(lengths):
            token = start if index % 2 == 0 else v.B if start == v.A else v.A
            active.extend([token] * (1 + added))
    gaps = neutral_gaps(size, length - 1, rng, spec.max_neutral_gap)
    tokens = [v.BOS] + insert_gaps(active, gaps, v.NEUTRAL)
    return example_from_targets(spec.task, tokens, oracle_targets(tokens, spec))
