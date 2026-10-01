"""Complete prime-field division data matching the published openai/grok format.

Source: Power et al.'s author implementation, openai/grok at commit
3d64b1d8c1d595dd8ebdb7771998823f1b14c7b3, grok/data.py. The reference
format supervises both the answer and EOS, with EOS also starting the prompt.
The independent inverse oracle checks the multiplication-based construction.
"""

from dataclasses import dataclass
import hashlib
import itertools
import math

import numpy as np


OPERATORS = ("+", "-", "*", "/", "**2+", "**3+", "x**2+y**2_mod_97",
             "x**2+y**2+x*y_mod_97", "x**2+y**2+x*y+x_mod_97", "x**3+x*y_mod_97",
             "x**3+x*y**2+y_mod_97", "(x._value//y)if(y._value%2==1)else(x-y)_mod_97",
             "s5", "s5conj", "s5aba", "+*", "+-", "sort", "reverse", "copy")


def vocabulary(prime=97):
    return ("<|eos|>", "=", *sorted(OPERATORS), *(str(number) for number in range(prime)),
            *("".join(map(str, word)) for word in itertools.permutations(range(5))))


def is_prime(number):
    return number >= 2 and all(number % divisor for divisor in range(2, math.isqrt(number) + 1))


def answer(numerator, denominator, prime):
    if not is_prime(prime) or not 0 <= numerator < prime or not 1 <= denominator < prime:
        raise ValueError("Division requires canonical values in a prime field and a nonzero denominator")
    return numerator * pow(denominator, -1, prime) % prime


def complete_rows(prime=97):
    if not is_prime(prime):
        raise ValueError("Composite-modulus division can have ambiguous or missing answers")
    tokens = vocabulary(prime)
    indices = {token: index for index, token in enumerate(tokens)}
    rows = []
    for quotient in range(prime):
        for denominator in range(1, prime):
            numerator = denominator * quotient % prime
            if answer(numerator, denominator, prime) != quotient:
                raise ValueError("Independent division oracle rejected the generated answer")
            rows.append((0, indices[str(numerator)], indices["/"], indices[str(denominator)],
                         indices["="], indices[str(quotient)], 0))
    return np.array(rows, dtype=np.int64)


def fingerprint(rows):
    return hashlib.sha256(np.asarray(rows, dtype="<i8").tobytes()).hexdigest()


@dataclass(frozen=True)
class ModularCorpus:
    prime: int
    train: np.ndarray
    heldout: np.ndarray
    tokens: tuple[str, ...]
    data_seed: int

    def summary(self):
        return {"prime": self.prime, "domain_size": self.prime * (self.prime - 1),
                "train_examples": len(self.train), "heldout_examples": len(self.heldout),
                "train_fraction": len(self.train) / (self.prime * (self.prime - 1)),
                "data_seed": self.data_seed, "vocab_size": len(self.tokens),
                "train_fingerprint": fingerprint(self.train), "heldout_fingerprint": fingerprint(self.heldout),
                "split_policy": "exhaustive_disjoint_two_way_author_protocol",
                "prompt_tokens": 5, "supervised_targets": 2,
                "scope": "all_unseen_operand_pairs_in_fixed_prime_field; no_length_OOD"}


def make_corpus(prime=97, train_fraction=.2, data_seed=0):
    if not 0 < train_fraction < 1:
        raise ValueError("Train fraction must lie strictly between zero and one")
    rows = complete_rows(prime)
    # Match the author's legacy NumPy shuffle, separate from model/batch RNG.
    np.random.RandomState(data_seed).shuffle(rows)
    count = round(len(rows) * train_fraction)
    if not 0 < count < len(rows):
        raise ValueError("Both exhaustive corpus partitions must be nonempty")
    return ModularCorpus(prime, rows[:count], rows[count:], vocabulary(prime), data_seed)

