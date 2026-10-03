"""Nested finite pools, optional input separation, and overlap diagnostics."""

from collections import Counter
from dataclasses import replace
import math

from .data import build_split
from .vocabulary import IGNORE


def problem_key(example):
    if example.task == "random_lm":
        return example.tokens  # Every random payload, rather than its shared prompt.
    return example.prompt or example.tokens


def separated_split(spec, name, seed, count, excluded=()):
    """Deterministic rejection sampling; fail if a small domain cannot supply a pool."""
    seen = set(excluded)
    selected = []
    budget, maximum = max(64, 2 * count), max(4096, 64 * count)
    consumed = 0
    while True:
        split = build_split(spec, name, seed, budget)
        for example in split.examples[consumed:]:
            key = problem_key(example)
            if key not in seen:
                selected.append(example)
                seen.add(key)
                if len(selected) == count:
                    return replace(split, examples=tuple(selected))
        consumed = budget
        if budget >= maximum:
            raise ValueError(f"Cannot obtain {count} disjoint {name} inputs; enlarge the task domain or reduce split sizes")
        budget = min(2 * budget, maximum)


def study_pool(spec, config):
    if spec.task == "random_lm" and config.split_policy != "independent":
        raise ValueError("The IID random control requires independent sampling with replacement")
    if config.split_policy == "independent":
        return {name: build_split(spec, name, config.data_seed, count) for name, count in
                (("train", config.train_examples), ("validation", config.validation_examples), ("test", config.test_examples))}
    train = separated_split(spec, "train", config.data_seed, config.train_examples)
    train_keys = {problem_key(row) for row in train.examples}
    validation = separated_split(spec, "validation", config.data_seed, config.validation_examples, train_keys)
    excluded = train_keys | {problem_key(row) for row in validation.examples}
    test = separated_split(spec, "test", config.data_seed, config.test_examples, excluded)
    return {"train": train, "validation": validation, "test": test}


def corpus_report(splits):
    keys = {name: Counter(problem_key(row) for row in split.examples) for name, split in splits.items()}
    sizes = {name: {"examples": sum(counts.values()), "unique_inputs": len(counts),
                    "duplicate_rows": sum(counts.values()) - len(counts)} for name, counts in keys.items()}
    overlaps = {}
    names = tuple(keys)
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            shared = keys[first].keys() & keys[second].keys()
            overlaps[f"{first}/{second}"] = {"unique_inputs": len(shared),
                                           "rows_in_first": sum(keys[first][key] for key in shared),
                                           "rows_in_second": sum(keys[second][key] for key in shared)}
    return {"splits": sizes, "overlaps": overlaps, "identity": "raw_prompt_or_prefix_input; random_lm_full_payload"}


def context_report(examples):
    """An empirical causal-conflict floor, not a claim about population risk."""
    contexts = {}
    for example in examples:
        for position, target in enumerate(example.targets):
            if target != IGNORE:
                key = example.tokens[:position + 1]
                contexts.setdefault(key, Counter())[target] += 1
    total = sum(sum(labels.values()) for labels in contexts.values())
    loss_sum = error_sum = conflicting = 0
    for labels in contexts.values():
        count = sum(labels.values())
        conflicting += len(labels) > 1
        error_sum += count - max(labels.values())
        loss_sum += sum(-n * math.log(n / count) for n in labels.values())
    return {"supervised_contexts": len(contexts), "conflicting_contexts": conflicting,
            "empirical_min_token_error": error_sum / total,
            "empirical_min_token_loss_nats": loss_sum / total,
            "scope": "any deterministic causal predictor on these observed contexts"}
