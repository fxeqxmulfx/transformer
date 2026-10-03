"""Finite random compositions and neutral gaps; no rejection sampling."""

import math
import sys


def distribute(total, parts, rng, cap=None):
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
    return distribute(length - size, size + 1, rng, maximum)


def insert_gaps(active, gaps, neutral):
    if len(gaps) != len(active) + 1:
        raise ValueError("Every active symbol needs a surrounding gap")
    result = [neutral] * gaps[0]
    for token, gap in zip(active, gaps[1:]):
        result.append(token)
        result.extend([neutral] * gap)
    return result


def weighted_positions(available, count, alpha, rng):
    # Exponential races implement weighted sampling without replacement.
    # Log-space scores avoid underflow for large distances or exponents.
    scores = [(math.log(-math.log(max(rng.random(), sys.float_info.min)))
               + alpha * math.log(position), position) for position in available]
    return [position for _, position in sorted(scores)[:count]]
