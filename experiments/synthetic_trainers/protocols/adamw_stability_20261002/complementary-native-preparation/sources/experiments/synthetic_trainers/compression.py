"""Sequence code-length proxies, not Shannon mutual information measurements.

arXiv:2505.24832v3, Sections 2.3 and 3.2: compare whole-sequence likelihoods
with the known uniform reference. The equal mixture is normalized; the
pointwise maximum only defines a clipped likelihood proxy.
"""

import math

import torch
from torch.nn import functional as F

from .records import collate
from .vocabulary import EOS, IGNORE


@torch.no_grad()
def code_lengths(model, examples, batch_size, device="cpu"):
    if not examples or batch_size < 1:
        raise ValueError("Code lengths require examples and a positive batch size")
    previous = model.training
    model.eval()
    scores = []
    try:
        for start in range(0, len(examples), batch_size):
            batch = collate(examples[start:start + batch_size], device)
            logits = model(batch.tokens).float()
            losses = F.cross_entropy(logits.transpose(1, 2), batch.targets,
                                     ignore_index=IGNORE, reduction="none") / math.log(2)
            if not torch.isfinite(losses).all():
                raise RuntimeError("Nonfinite sequence code length")
            valid = batch.targets != IGNORE
            payload = valid & (batch.targets != EOS)
            for total, content, count in zip(losses.sum(1).tolist(), (losses * payload).sum(1).tolist(), valid.sum(1).tolist()):
                scores.append({"bits": total, "payload_bits": content, "targets": count,
                               "bits_per_target": total / count})
        return scores
    finally:
        model.train(previous)


def uniform_compression(examples, scores, symbols, parameters):
    if len(examples) != len(scores) or not examples or symbols < 2 or parameters < 1:
        raise ValueError("Need matching examples and code lengths, symbols >= 2, and positive parameters")
    if any(row.task != "random_lm" for row in examples):
        raise ValueError("The uniform entropy reference applies only to random_lm")
    reference = [(len(row.answer) - 1) * math.log2(symbols) for row in examples]
    target = [score["bits"] for score in scores]
    if any(not math.isfinite(bits) or bits < 0 for bits in target):
        raise ValueError("Code lengths must be finite and nonnegative")
    if any(not math.isfinite(score["payload_bits"]) or not 0 <= score["payload_bits"] <= score["bits"] for score in scores):
        raise ValueError("Payload code length must lie between zero and the total")
    # -log2((2^-a + 2^-b)/2), computed without underflow.
    mixture = [min(a, b) + 1 - math.log2(1 + 2 ** (-abs(a - b))) for a, b in zip(reference, target)]
    gain = sum(a - b for a, b in zip(reference, target))
    clipped = sum(max(0.0, a - b) for a, b in zip(reference, target))
    mixture_gain = sum(a - b for a, b in zip(reference, mixture))
    return {"reference_entropy_bits": sum(reference), "model_code_bits": sum(target),
            "payload_code_bits": sum(score["payload_bits"] for score in scores),
            "eos_code_bits": sum(score["bits"] - score["payload_bits"] for score in scores),
            "net_gain_bits": gain, "net_bits_per_parameter": gain / parameters,
            "clipped_sequence_gain_bits": clipped,
            "mixture_code_bits": sum(mixture), "mixture_gain_bits": mixture_gain,
            "mixture_bits_per_parameter": mixture_gain / parameters,
            "scope": "finite_dataset_likelihood_proxy; conditional_on_lengths; not_mutual_information"}


def membership_auc(member_scores, nonmember_scores):
    """Loss-only rank AUC: lower mean answer loss predicts membership, ties split."""
    if not member_scores or not nonmember_scores:
        raise ValueError("Membership AUC requires both populations")
    if any(not math.isfinite(score["bits_per_target"]) for score in (*member_scores, *nonmember_scores)):
        raise ValueError("Membership losses must be finite")
    ranked = sorted((score["bits_per_target"], kind) for kind, scores in
                    ((True, member_scores), (False, nonmember_scores)) for score in scores)
    wins = nonmembers_below = 0.0
    i = 0
    while i < len(ranked):
        j = i + 1
        while j < len(ranked) and ranked[j][0] == ranked[i][0]:
            j += 1
        members = sum(kind for _, kind in ranked[i:j])
        nonmembers = j - i - members
        # Count higher-loss nonmembers for every member in this tie group.
        wins += members * (len(nonmember_scores) - nonmembers_below - nonmembers / 2)
        nonmembers_below += nonmembers
        i = j
    return wins / (len(member_scores) * len(nonmember_scores))


def prefix_extraction(model, examples, batch_size, device="cpu", fractions=(0.25, 0.5, 0.75)):
    """Greedy suffix extraction from true partial prefixes (Section 4 adaptation)."""
    from .generation import rollout

    if not examples or batch_size < 1 or any(row.task != "random_lm" for row in examples):
        raise ValueError("Prefix extraction requires random_lm examples and a positive batch size")
    if not fractions or any(not math.isfinite(fraction) or not 0 <= fraction < 1 for fraction in fractions):
        raise ValueError("Prefix fractions must be finite and in [0, 1)")
    results = []
    for fraction in fractions:
        exact = correct = total = prefix_tokens = 0
        for start in range(0, len(examples), batch_size):
            rows = examples[start:start + batch_size]
            lengths = [min(len(row.answer) - 2, math.floor((len(row.answer) - 1) * fraction)) for row in rows]
            prompts = [row.prompt + row.answer[:length] for row, length in zip(rows, lengths)]
            references = [row.answer[length:] for row, length in zip(rows, lengths)]
            generated = rollout(model, prompts, [len(answer) for answer in references], device)
            for predicted, reference, length in zip(generated.predictions, references, lengths):
                exact += tuple(predicted) == reference
                correct += sum(i < len(predicted) and predicted[i] == token for i, token in enumerate(reference))
                total += len(reference)
                prefix_tokens += length
        results.append({"prefix_fraction": fraction, "provided_payload_tokens": prefix_tokens,
                        "suffix_exact_accuracy": exact / len(examples), "suffix_token_accuracy": correct / total,
                        "examples": len(examples), "suffix_target_tokens": total,
                        "scope": "true_partial_prefix; at_least_one_unseen_payload_token; greedy_suffix_includes_EOS"})
    return results
