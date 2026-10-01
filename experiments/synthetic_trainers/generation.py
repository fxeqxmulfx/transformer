"""Greedy causal rollout using prompts alone, with explicit termination limits."""

from dataclasses import dataclass
import time

import torch

from .runtime import synchronize
from .vocabulary import EOS, PAD


@dataclass(frozen=True)
class Rollout:
    predictions: tuple[tuple[int, ...], ...]
    terminated: tuple[bool, ...]
    forward_calls: int
    padded_tokens_processed: int
    attention_cells_processed: int
    seconds: float


@torch.no_grad()
def rollout(model, prompts, limits, device="cpu"):
    """Right padding is causal; read the last real position of each active row.

    Limits come from prompt structure. Rows stop on EOS or their own bound;
    generated symbols, including mistakes, become the next step's context.
    There is no token filtering or replacement with oracle output.
    """
    if not prompts or len(prompts) != len(limits) or any(not prompt for prompt in prompts):
        raise ValueError("Need nonempty prompts and one limit per prompt")
    if any(limit < 1 for limit in limits):
        raise ValueError("Generation limits must be positive")
    contexts = [list(prompt) for prompt in prompts]
    predictions = [[] for _ in prompts]
    active = list(range(len(prompts)))
    previous = model.training
    model.eval()
    calls = processed = cells = 0
    synchronize(device)
    started = time.perf_counter()
    try:
        while active:
            sizes = [len(contexts[row]) for row in active]
            length = max(sizes)
            tokens = torch.full((len(active), length), PAD, dtype=torch.long, device=device)
            for offset, row in enumerate(active):
                tokens[offset, :sizes[offset]] = torch.tensor(contexts[row], device=device)
            logits = model(tokens)
            last = logits[torch.arange(len(active), device=device), torch.tensor(sizes, device=device) - 1]
            if not torch.isfinite(last).all():
                raise RuntimeError("Nonfinite generation logits")
            selected = last.argmax(-1).tolist()
            calls += 1
            processed += tokens.numel()
            cells += len(active) * length * length
            remaining = []
            for row, token in zip(active, selected):
                predictions[row].append(token)
                if token != EOS and len(predictions[row]) < limits[row]:
                    contexts[row].append(token)
                    remaining.append(row)
            active = remaining
        synchronize(device)
        return Rollout(tuple(tuple(row) for row in predictions),
                       tuple(row[-1] == EOS for row in predictions), calls, processed, cells,
                       time.perf_counter() - started)
    finally:
        model.train(previous)
