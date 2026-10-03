"""Splits: finite samples of a generator, each from a seed of its own, and their fingerprints.

A port of `experiments/synthetic_trainers/data.py` and `corpus.py`. The rows
of a split are a prefix of an endless stream seeded by the problem, the data
seed and the split's name: a smaller split is a prefix of a larger one, and
the formats of one problem share their problems.
"""

from dataclasses import dataclass, replace
import hashlib
from itertools import islice
import json
import random

from . import generator
from .base import Generator
from .noise import noisy_split
from .records import Example

GENERATOR_VERSION = 2
POOLS = ("train", "validation", "test")


@dataclass(frozen=True)
class Split:
    name: str
    generator: Generator
    seed: int
    examples: tuple[Example, ...]

    @property
    def fingerprint(self):
        payload = {"version": GENERATOR_VERSION, "name": self.name, "spec": self.generator.identity(),
                   "seed": self.seed, "examples": [vars(example) for example in self.examples]}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def stream(source, name, seed):
    """Every row a split `name` of `source` can hold, in order, each checked by the oracle."""
    if name not in POOLS or seed < 0:
        raise ValueError("Need a valid split and a nonnegative seed")
    identity = json.dumps({"version": GENERATOR_VERSION, "spec": source.problem_identity(),
                           "seed": seed, "split": name}, sort_keys=True)
    rng = random.Random(int.from_bytes(hashlib.sha256(identity.encode()).digest(), "big"))
    while True:
        for example in source.draw(rng):
            source.validate(example)
            yield example


def build_split(source, name, seed, count):
    if count < 1:
        raise ValueError("A split holds at least one example")
    return Split(name, source, seed, tuple(islice(stream(source, name, seed), count)))


def separated_split(source, name, seed, count, excluded=()):
    """The first `count` rows of the stream whose inputs are new, and not `excluded`.

    It reads at most max(4096, 64 count) rows, and fails if a small domain
    cannot supply them.
    """
    seen, selected = set(excluded), []
    for example in islice(stream(source, name, seed), max(4096, 64 * count)):
        key = source.key(example)
        if key not in seen:
            seen.add(key)
            selected.append(example)
            if len(selected) == count:
                return Split(name, source, seed, tuple(selected))
    raise ValueError(f"Cannot obtain {count} disjoint {name} inputs; enlarge the task domain or reduce split sizes")


def pools(source, seed, sizes, disjoint):
    """The train, validation and test pools; with `disjoint`, no input recurs within or across them."""
    if not disjoint:
        return {name: build_split(source, name, seed, sizes[name]) for name in POOLS}
    result, seen = {}, set()
    for name in POOLS:
        result[name] = separated_split(source, name, seed, sizes[name], seen)
        seen |= {source.key(example) for example in result[name].examples}
    return result


def benchmark_splits(benchmark, seed):
    """Every split of a `Synthetic` benchmark at data seed `seed`, and the report of its label noise.

    `train`, `validation` and `test` sample the training distribution, and
    `test/P` each held-out distribution P. A study adds `train_clean`, the
    training rows with the oracle's labels, whose noisy copy `train` is, and
    `validation/P` (`training.train_run` and `studies.py`); `select` adds
    `validation/P` of the selected P alone.
    """
    study = benchmark.study
    source = generator(benchmark.problem)
    sizes = {"train": benchmark.train, "validation": benchmark.validation, "test": benchmark.test}
    noise = None
    if study is None:
        splits = pools(source, seed, sizes, disjoint=False)
    else:
        splits = pools(source, seed, {**sizes, "train": study.pool or benchmark.train}, study.disjoint)
        clean = replace(splits["train"], examples=splits["train"].examples[:benchmark.train])
        splits["train"], noise = noisy_split(clean, study.noise, study.noise_seed)
        splits["train_clean"] = clean
    for name, problem in benchmark.probes.items():
        held_out = generator(problem)
        if study is not None or name == benchmark.select:
            splits[f"validation/{name}"] = build_split(held_out, "validation", seed, benchmark.validation)
        splits[f"test/{name}"] = build_split(held_out, "test", seed, benchmark.test)
    return splits, noise
