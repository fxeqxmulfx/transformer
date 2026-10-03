"""Frozen uniform incorrect-label noise, adapted from arXiv:1912.02292v1, Section 4.

Noise is assigned once per causal input (or prompt and answer position), rather
than resampled per epoch. Repeated deterministic problems retain the same labels.
Structural EOS, BOS, separators, and index hints are never corrupted.
"""

from dataclasses import replace
import hashlib
import json
import math
import random

from .records import example_from_targets, generation_example
from . import vocabulary as v


def label_domain(example, position, spec):
    """Legal content labels; an empty domain identifies a structural target."""
    target = example.targets[position]
    if target in (v.IGNORE, v.BOS, v.EOS, v.COMMA, v.SEP):
        return ()
    if spec.task in ("dyck", "dyck2"):
        return (v.BALANCED, v.INCOMPLETE, v.INVALID)
    if spec.task in ("blocks", "crasp", "boolean_and"):
        return (v.REJECT, v.ACCEPT)
    if spec.task == "parity":
        return (v.EVEN, v.ODD) if target in (v.EVEN, v.ODD) else ()
    if spec.task == "addition":
        return range(v.DIGIT_BASE, v.DIGIT_BASE + 10) if v.DIGIT_BASE <= target < v.DIGIT_BASE + 10 else ()
    if spec.task == "mqar":
        return range(v.IDENTITY_BASE + spec.symbols, v.IDENTITY_BASE + 2 * spec.symbols)
    if spec.task in ("histogram", "histogram2", "mode", "count") and target >= spec.number_base:
        from .sequence_oracles import generation_problem_size

        maximum = spec.number_limit if spec.task == "count" else generation_problem_size(example.prompt, spec)
        return range(spec.number_base + 1, spec.number_base + maximum + 1)
    return range(v.IDENTITY_BASE, spec.number_base)


def corrupt_example(example, spec, rate, seed):
    if not math.isfinite(rate) or not 0 <= rate <= 1 or seed < 0:
        raise ValueError("Need noise in [0, 1] and a nonnegative seed")
    if spec.task == "random_lm" and rate:
        raise ValueError("The IID random control already has random targets; label noise is unsupported")
    labels = list(example.targets)
    eligible = changed = 0
    for position, target in enumerate(example.targets):
        domain = label_domain(example, position, spec)
        if len(domain) < 2:
            continue
        if target not in domain:
            raise ValueError("A content target is outside its noise domain")
        eligible += 1
        if not rate:
            continue
        # Generated labels use the original prompt and position, without the
        # current label or preceding corrupted labels in the random seed.
        address = (example.prompt, position - len(example.prompt) + 1) if example.prompt else example.tokens[:position + 1]
        identity = json.dumps([seed, spec.task, address, list(domain)])
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
    """Keep the oracle-validated corpus separate from the observed training labels."""
    rows, eligible, changed = [], 0, 0
    for example in clean.examples:
        observed, candidates, flips = corrupt_example(example, clean.spec, rate, seed)
        rows.append(observed)
        eligible += candidates
        changed += flips
    return replace(clean, examples=tuple(rows)), {
        "kernel": "uniform_incorrect_label", "rate": rate, "seed": seed,
        "eligible_targets": eligible, "changed_targets": changed,
        "realized_rate": changed / eligible if eligible else 0.0,
        "assignment": "fixed_causal_input_or_prompt_position",
    }


def noise_fit_metrics(model, observed, clean, batch_size, device="cpu"):
    """Accuracy on corrupted positions, with the observed preceding answers."""
    import torch

    from .records import collate

    correct_noise = correct_clean = count = 0
    previous = model.training
    model.eval()
    try:
        with torch.no_grad():
            for start in range(0, len(clean), batch_size):
                noisy_batch = collate(observed[start:start + batch_size], device)
                clean_batch = collate(clean[start:start + batch_size], device)
                mask = noisy_batch.targets != clean_batch.targets
                logits = model(noisy_batch.tokens)
                if not torch.isfinite(logits).all():
                    raise RuntimeError("Nonfinite noise diagnostic logits")
                predictions = logits.argmax(-1)
                count += int(mask.sum())
                correct_noise += int(((predictions == noisy_batch.targets) & mask).sum())
                correct_clean += int(((predictions == clean_batch.targets) & mask).sum())
        return {"corrupted_targets": count, "observed_label_accuracy": correct_noise / count if count else None,
                "clean_label_accuracy": correct_clean / count if count else None,
                "conditioning": "observed_inputs_and_preceding_corrupted_answers"}
    finally:
        model.train(previous)
