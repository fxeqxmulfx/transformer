"""Immutable examples, with their supervision apart from their input.

A port of `experiments/synthetic_trainers/records.py`. The fields are the
historical ones, which a split's fingerprint hashes.
"""

from dataclasses import dataclass

from .vocabulary import EOS, IGNORE, SEP


@dataclass(frozen=True)
class Example:
    """An input, its target at every position (IGNORE where unsupervised), and where the target changes.

    A generated-answer example also holds its prompt and answer: the input is
    the prompt and the answer without its final EOS, and every answer token
    is the target of the position before it. `generation_limit` bounds a
    free generation; it depends on the prompt alone.
    """
    task: str
    tokens: tuple[int, ...]
    targets: tuple[int, ...]
    changes: tuple[bool, ...]
    prompt: tuple[int, ...] = ()
    answer: tuple[int, ...] = ()
    generation_limit: int = 0

    def __post_init__(self):
        if not self.tokens or not len(self.tokens) == len(self.targets) == len(self.changes):
            raise ValueError("Example fields must have the same nonzero length")
        if all(target == IGNORE for target in self.targets):
            raise ValueError("An example needs at least one supervised answer")
        if any(change and target == IGNORE for change, target in zip(self.changes, self.targets)):
            raise ValueError("A change must be at a supervised position")
        if bool(self.prompt) != bool(self.answer):
            raise ValueError("Generation requires both a prompt and an answer")
        if self.prompt:
            if self.prompt[-1] != SEP or self.answer[-1] != EOS or EOS in self.answer[:-1]:
                raise ValueError("Generation requires a separator and one final EOS")
            if self.tokens != self.prompt + self.answer[:-1]:
                raise ValueError("Generation inputs must contain only preceding answer tokens")
            if self.targets != (IGNORE,) * (len(self.prompt) - 1) + self.answer:
                raise ValueError("Generation targets must be shifted by one token")
            if not len(self.answer) <= self.generation_limit:
                raise ValueError("Generation limit cannot truncate the reference answer")
        elif self.generation_limit:
            raise ValueError("Prefix examples cannot have a generation limit")


def example_from_targets(task, tokens, targets, **fields):
    """An example whose changes are the supervised targets that differ from the previous supervised one."""
    previous = None
    changes = []
    for target in targets:
        changes.append(target != IGNORE and (previous is None or target != previous))
        if target != IGNORE:
            previous = target
    return Example(task, tuple(tokens), tuple(targets), tuple(changes), **fields)


def generation_example(task, prompt, answer, limit):
    """Teacher forcing predicts each token before it is present in the input."""
    prompt, answer = tuple(prompt), tuple(answer)
    return example_from_targets(task, prompt + answer[:-1], (IGNORE,) * (len(prompt) - 1) + answer,
                                prompt=prompt, answer=answer, generation_limit=limit)
