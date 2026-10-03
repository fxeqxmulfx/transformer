"""Immutable examples and right-padded batches with separate supervision."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

from .vocabulary import EOS, IGNORE, PAD, SEP


@dataclass(frozen=True)
class Example:
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
    return example_from_targets(task, prompt + answer[:-1],
                                (IGNORE,) * (len(prompt) - 1) + answer,
                                prompt=prompt, answer=answer, generation_limit=limit)


@dataclass
class Batch:
    tokens: "torch.Tensor"
    targets: "torch.Tensor"
    changes: "torch.Tensor"


def collate(examples, device="cpu"):
    import torch

    if not examples:
        raise ValueError("Cannot collate an empty batch")
    length = max(len(example.tokens) for example in examples)
    tokens = torch.full((len(examples), length), PAD, dtype=torch.long)
    targets = torch.full_like(tokens, IGNORE)
    changes = torch.zeros_like(tokens, dtype=torch.bool)
    for row, example in enumerate(examples):
        size = len(example.tokens)
        tokens[row, :size] = torch.tensor(example.tokens)
        targets[row, :size] = torch.tensor(example.targets)
        changes[row, :size] = torch.tensor(example.changes)
    return Batch(tokens.to(device), targets.to(device), changes.to(device))
