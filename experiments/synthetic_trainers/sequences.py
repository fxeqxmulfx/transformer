"""Construct RASP/RASP-L problems, then supervise independently parsed answers."""

from collections import Counter
import random

from . import vocabulary as v
from .arithmetic import addition_prompt
from .records import generation_example
from .sequence_oracles import generation_answer, generation_limit


def symbol_word(spec, rng, size):
    alphabet = range(v.IDENTITY_BASE, v.IDENTITY_BASE + spec.symbols)
    if spec.task in ("copy", "reverse", "sort") and spec.unique:
        return rng.sample(alphabet, size)
    maximum = 2 if spec.task in ("copy", "reverse") else 5 if spec.task == "mode" else 8
    pool = rng.sample(alphabet, rng.randint(2, min(spec.symbols, maximum, max(2, size))))
    word = [rng.choice(pool) for _ in range(size)]
    if spec.task == "mode":
        counts = Counter(word)
        highest = max(counts.values())
        winners = [x for x in counts if counts[x] == highest]
        if len(winners) > 1:
            # Move one occurrence to the selected winner; its count is now uniquely highest.
            winner = rng.choice(winners)
            donor = rng.choice([x for x in winners if x != winner])
            positions = [i for i, x in enumerate(word) if x == donor]
            word[rng.choice(positions)] = winner
    return word


def bit_prompt(spec, rng, size, hint_rng=None):
    if spec.task == "boolean_and":
        bits = [v.ONE] * size
        if rng.random() < 0.5:
            cut = size - max(1, size // 4)
            positions = (range(cut) if spec.and_region == "early" else range(cut, size)) if spec.and_shift else range(size)
            bits[rng.choice(positions)] = v.ZERO
    else:
        bits = [rng.choice((v.ZERO, v.ONE)) for _ in range(size)]
    prompt = [v.BOS]
    start = (hint_rng or rng).randint(0, spec.number_limit - size) if spec.task == "parity" and spec.index_hints else 0
    for index, bit in enumerate(bits):
        if spec.task == "parity" and spec.index_hints:
            prompt.append(spec.number_base + start + index)
        prompt.append(bit)
    return (*prompt, v.SEP)


def sequence_example(spec, rng, size):
    # Always consume one independent format seed, even without positional hints.
    hint_rng = random.Random(rng.getrandbits(128))
    if spec.task == "addition":
        prompt = addition_prompt(spec, rng, size, hint_rng)
    elif spec.task in ("parity", "boolean_and"):
        prompt = bit_prompt(spec, rng, size, hint_rng)
    elif spec.task == "count":
        start = rng.randint(1, spec.number_limit - size + 1)
        prompt = (v.BOS, spec.number_base + start, spec.number_base + start + size - 1, v.SEP)
    else:
        bos = spec.histogram_bos if spec.task == "histogram" else True
        prompt = (*([v.BOS] if bos else []), *symbol_word(spec, rng, size), v.SEP)
    return generation_example(spec.task, prompt, generation_answer(prompt, spec), generation_limit(prompt, spec))
