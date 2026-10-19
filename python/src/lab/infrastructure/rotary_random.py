"""The browser lab's SplitMix64 and Box-Muller draws.

Adapted from grokking_abmod/web/own-channel/python/rotate90.py at
f49969393a7482490d45483d8a4fd3046b1adede (MIT; third_party/rotate90/NOTICE).
The RNG and draw order are unchanged. Sizes and seeds become arguments so
the lab can test other widths and primes with the same protocol.
"""

import math

MASK64 = (1 << 64) - 1


class SplitMix64:
    def __init__(self, seed):
        self.state = seed & MASK64

    def next(self):
        self.state = (self.state + 0x9E3779B97F4A7C15) & MASK64
        value = self.state
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
        return value ^ (value >> 31)

    def unit(self):
        return (self.next() >> 40) / 16777216

    def index(self, bound):
        if bound <= 0:
            raise ValueError("A random index needs a positive bound")
        threshold = (1 << 64) % bound
        while True:
            value = self.next()
            if value >= threshold:
                return value % bound

    def shuffle(self, values):
        for end in range(len(values) - 1, 0, -1):
            index = self.index(end + 1)
            values[end], values[index] = values[index], values[end]


def normal_values(random, count, variance):
    """Consecutive Box-Muller pairs; Rotate90's even widths make every count even."""
    if count % 2:
        raise ValueError("The browser initializer consumes whole Box-Muller pairs")
    deviation, values = math.sqrt(variance), []
    for _ in range(0, count, 2):
        radius = math.sqrt(-2 * math.log(1 - random.unit()))
        angle = 2 * math.pi * random.unit()
        values.extend((deviation * radius * math.cos(angle), deviation * radius * math.sin(angle)))
    return values
