"""Uniform IID sequence control: arXiv:2505.24832v3, Section 3.2.

The length is supplied, the payload consists of independent uniform symbols,
and EOS is deterministic. Conditional on lengths, entropy is sum(S) log2(V).
There is deliberately no deterministic answer oracle for a random payload.
"""

from .records import generation_example
from . import vocabulary as v


def random_example(spec, rng, length):
    prompt = (v.BOS, spec.number_base + length, v.SEP)
    answer = tuple(rng.randrange(v.IDENTITY_BASE, spec.number_base) for _ in range(length))
    return generation_example(spec.task, prompt, (*answer, v.EOS), length + 1)


def requested_length(prompt, spec):
    if len(prompt) != 3 or prompt[0] != v.BOS or prompt[-1] != v.SEP:
        raise ValueError("Random control requires BOS, requested length, and SEP")
    length = prompt[1] - spec.number_base
    if not (spec.min_length or spec.length) <= length <= spec.length:
        raise ValueError("Random payload length is outside the requested range")
    return length


def validate_random(example, spec):
    length = requested_length(example.prompt, spec)
    if len(example.answer) != length + 1 or example.generation_limit != length + 1:
        raise ValueError("Random payload or generation bound has the wrong length")
    if any(not v.IDENTITY_BASE <= token < spec.number_base for token in example.answer[:-1]):
        raise ValueError("Random payload token is outside its uniform alphabet")
    if example.task != spec.task or len(example.tokens) > spec.context_length:
        raise ValueError("Random example task or context does not match its specification")
    # Example already checks the shift and EOS; check the change mask too.
    expected = generation_example(spec.task, example.prompt, example.answer, length + 1)
    if example != expected:
        raise ValueError("Invalid random example supervision")
