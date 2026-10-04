"""Finite random compositions and neutral gaps, without rejection sampling.

A port of `experiments/synthetic_trainers/sampling.py`; every draw from the
generator is the historical one.
"""

import math
import sys


def distribute(total, parts, rng, cap=None):
    """`total` units over `parts` parts, one uniform part at a time, none above `cap`."""
    if total < 0 or parts < 1 or (cap is not None and total > parts * cap):
        raise ValueError("Impossible composition")
    result = [0] * parts
    available = list(range(parts)) if cap != 0 else []
    for _ in range(total):
        offset = rng.randrange(len(available))
        index = available[offset]
        result[index] += 1
        if cap is not None and result[index] == cap:
            available[offset] = available[-1]
            available.pop()
    return result


def neutral_gaps(size, length, rng, maximum=None):
    """The neutral tokens before, between and after `size` active symbols filling `length`."""
    return distribute(length - size, size + 1, rng, maximum)


def insert_gaps(active, gaps, neutral):
    if len(gaps) != len(active) + 1:
        raise ValueError("Every active symbol needs a surrounding gap")
    result = [neutral] * gaps[0]
    for token, gap in zip(active, gaps[1:]):
        result.append(token)
        result.extend([neutral] * gap)
    return result


def uniform_draws(rng, start, stop, count):
    """`count` draws of `rng.randrange(start, stop)`, the same values from the same stream, at a third of the cost.

    CPython's `randrange` draws `getrandbits(n.bit_length())` until it falls
    below n = stop - start (`Random._randbelow_with_getrandbits`); so does
    this, without the two calls around each draw.
    """
    width = stop - start
    if width < 1:
        raise ValueError("Need a nonempty range")
    bits, getrandbits = width.bit_length(), rng.getrandbits
    draws = []
    for _ in range(count):
        draw = getrandbits(bits)
        while draw >= width:
            draw = getrandbits(bits)
        draws.append(start + draw)
    return draws


def weighted_positions(available, count, alpha, rng):
    """`count` of the positions, without replacement, with weight position^-alpha."""
    # Exponential races implement weighted sampling without replacement.
    # Log-space scores avoid underflow for large distances or exponents.
    log, random, tiny = math.log, rng.random, sys.float_info.min
    scores = [(log(-log(max(random(), tiny))) + alpha * log(position), position) for position in available]
    return [position for _, position in sorted(scores)[:count]]
