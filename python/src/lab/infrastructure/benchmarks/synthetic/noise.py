"""Label noise: each eligible training label replaced, with one probability, by another legal label.

A port of `experiments/synthetic_trainers/label_noise.py`, adapted from
arXiv:1912.02292v1, Section 4. Noise is drawn once per input and position,
from a seed of its own, not per epoch: a problem that recurs keeps its
labels, and a generated answer is addressed by its prompt and position,
whatever labels precede it. Structural targets (BOS, EOS, separators,
index hints) keep their labels.
"""

from dataclasses import replace
import hashlib
import json
import random

from .records import example_from_targets, generation_example


def corrupt(generator, example, rate, seed):
    """An example with noisy labels, how many of its labels could change, and how many did."""
    labels = list(example.targets)
    eligible = changed = 0
    for position, target in enumerate(example.targets):
        domain = generator.labels(example, position)
        if len(domain) < 2:
            continue
        if target not in domain:
            raise ValueError("A content target is outside its noise domain")
        eligible += 1
        if not rate:
            continue
        # Generated labels use the original prompt and position, without the
        # current label or preceding corrupted labels in the random seed.
        address = ((example.prompt, position - len(example.prompt) + 1) if example.prompt
                   else example.tokens[:position + 1])
        identity = json.dumps([seed, generator.name, address, list(domain)])
        rng = random.Random(int.from_bytes(hashlib.sha256(identity.encode()).digest(), "big"))
        if rng.random() < rate:
            index = rng.randrange(len(domain) - 1)
            index += index >= domain.index(target)
            labels[position] = domain[index]
            changed += 1
    if example.prompt:
        answer = labels[len(example.prompt) - 1:]
        observed = generation_example(example.task, example.prompt, answer, example.generation_limit)
    else:
        observed = example_from_targets(example.task, example.tokens, labels)
    return observed, eligible, changed


def noisy_split(clean, rate, seed):
    """The observed training labels, kept apart from the oracle's, and a report of the noise."""
    rows, eligible, changed = [], 0, 0
    for example in clean.examples:
        observed, candidates, flips = corrupt(clean.generator, example, rate, seed)
        rows.append(observed)
        eligible += candidates
        changed += flips
    return replace(clean, examples=tuple(rows)), {
        "kernel": "uniform_incorrect_label", "rate": rate, "seed": seed,
        "eligible_targets": eligible, "changed_targets": changed,
        "realized_rate": changed / eligible if eligible else 0.0,
        "assignment": "fixed_causal_input_or_prompt_position",
    }
