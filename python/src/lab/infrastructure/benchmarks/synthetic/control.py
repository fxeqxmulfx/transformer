"""A memorization control: uniform independent symbols after a prompt that states their count.

A port of `experiments/synthetic_trainers/random_control.py`. Source:
arXiv:2505.24832v3, Section 3.2, whose sequences have one length and no
prompt. Given the lengths, a split holds sum(length) log2(symbols) bits; no
oracle can answer a prompt.
"""

from .base import Answered
from .records import generation_example
from .vocabulary import BOS, EOS, IDENTITY_BASE, SEP


class RandomLM(Answered):
    name = "random_lm"
    formats = ("number_limit",)

    def controls(self):
        return {"symbols": self.task.symbols, "number_limit": self.task.number_limit}

    def draw(self, rng):
        # Single rows keep variable-length sampling independent, without paired lengths.
        length = rng.randint(self.minimum, self.length)
        prompt = (BOS, self.number_base + length, SEP)
        answer = tuple(rng.randrange(IDENTITY_BASE, self.number_base) for _ in range(length))
        return (generation_example(self.name, prompt, (*answer, EOS), length + 1),)

    def problem_size(self, prompt):
        if len(prompt) != 3 or prompt[0] != BOS or prompt[-1] != SEP:
            raise ValueError("Random control requires BOS, requested length, and SEP")
        length = prompt[1] - self.number_base
        if not self.minimum <= length <= self.length:
            raise ValueError("Random payload length is outside the requested range")
        return length

    def targets(self, tokens):
        raise ValueError("Random payloads have no deterministic raw-prompt oracle")

    def validate(self, example):
        length = self.problem_size(example.prompt)
        if len(example.answer) != length + 1 or example.generation_limit != length + 1:
            raise ValueError("Random payload or generation bound has the wrong length")
        if any(not IDENTITY_BASE <= token < self.number_base for token in example.answer[:-1]):
            raise ValueError("Random payload token is outside its uniform alphabet")
        if example.task != self.name or len(example.tokens) > self.context:
            raise ValueError("Random example task or context does not match its specification")
        # Example already checks the shift and EOS; check the change mask too.
        if example != generation_example(self.name, example.prompt, example.answer, length + 1):
            raise ValueError("Invalid random example supervision")

    def key(self, example):
        return example.tokens  # Every random payload, rather than its shared prompt.
