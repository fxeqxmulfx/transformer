"""Code lengths under a model: compression of random answers, membership, and extraction of their suffixes.

A port of `experiments/synthetic_trainers/compression.py`, after
arXiv:2505.24832v3, Sections 2.3, 3.2 and 4 (not in `papers/`): whole-answer
likelihoods compared with the uniform code that random answers have. They
are proxies on a finite split, not measurements of mutual information. A
code length that is not finite is None here; the historical one raised.
"""

import math

import torch
from torch.nn import functional as F

from .generation import rollout, score
from .rows import answers
from .vocabulary import EOS, IGNORE


@torch.no_grad()
def code_lengths(model, rows, batch):
    """Each example's code length in bits: of all its targets, of its payload before EOS, and per target.

    The model reads the rows (`Rows`) in chunks of `batch`; None if a code
    length is not finite.
    """
    model.eval()
    scores = []
    for start in range(0, len(rows), batch):
        tokens, targets, _ = rows[start:start + batch].rows.unbind(1)
        logits = model(tokens).float()
        losses = F.cross_entropy(logits.transpose(1, 2), targets, ignore_index=IGNORE,
                                 reduction="none") / math.log(2)
        if not torch.isfinite(losses).all():
            return None
        valid = targets != IGNORE
        payload = valid & (targets != EOS)
        for total, content, count in zip(losses.sum(1).tolist(), (losses * payload).sum(1).tolist(),
                                         valid.sum(1).tolist()):
            scores.append({"bits": total, "payload_bits": content, "targets": count,
                           "bits_per_target": total / count})
    return scores


def uniform_compression(examples, scores, symbols, parameters):
    """The bits a model's code saves on random answers over their uniform code, in all and per parameter.

    A random answer of n symbols has the uniform code n log2(symbols) bits,
    given its length. The equal mixture of the two codes is a code too; the
    gain of each sequence clipped at zero is a proxy only.
    """
    reference = [(len(example.answer) - 1) * math.log2(symbols) for example in examples]
    target = [entry["bits"] for entry in scores]
    # -log2((2^-a + 2^-b) / 2), without underflow.
    mixture = [min(a, b) + 1 - math.log2(1 + 2 ** (-abs(a - b))) for a, b in zip(reference, target)]
    gain = sum(a - b for a, b in zip(reference, target))
    mixture_gain = sum(a - b for a, b in zip(reference, mixture))
    return {"reference_entropy_bits": sum(reference), "model_code_bits": sum(target),
            "payload_code_bits": sum(entry["payload_bits"] for entry in scores),
            "eos_code_bits": sum(entry["bits"] - entry["payload_bits"] for entry in scores),
            "net_gain_bits": gain, "net_bits_per_parameter": gain / parameters,
            "clipped_sequence_gain_bits": sum(max(0.0, a - b) for a, b in zip(reference, target)),
            "mixture_code_bits": sum(mixture), "mixture_gain_bits": mixture_gain,
            "mixture_bits_per_parameter": mixture_gain / parameters,
            "scope": "finite_dataset_likelihood_proxy; conditional_on_lengths; not_mutual_information"}


def membership_auc(members, nonmembers):
    """The probability that a member's bits per target are below a nonmember's, ties counting half."""
    ranked = sorted((entry["bits_per_target"], member) for member, entries in ((True, members), (False, nonmembers))
                    for entry in entries)
    wins = below = 0.0
    start = 0
    while start < len(ranked):
        stop = start + 1
        while stop < len(ranked) and ranked[stop][0] == ranked[start][0]:
            stop += 1
        tied = sum(member for _, member in ranked[start:stop])
        others = stop - start - tied
        # Every member of a tie wins against each nonmember of a higher loss, and half of each tied one.
        wins += tied * (len(nonmembers) - below - others / 2)
        below += others
        start = stop
    return wins / (len(members) * len(nonmembers))


@torch.no_grad()
def prefix_extraction(model, examples, batch, device, fractions=(0.25, 0.5, 0.75)):
    """How much of each random answer greedy generation recovers from a true partial prefix of it.

    At each fraction f an answer of n symbols and EOS gives away its first
    min(n - 1, floor(n f)) symbols, so at least one is left to generate, and
    the rest, EOS included, is generated in chunks of `batch` (`rollout`).
    """
    model.eval()
    results = []
    for fraction in fractions:
        exact = correct = total = provided = 0
        for start in range(0, len(examples), batch):
            rows = examples[start:start + batch]
            lengths = [min(len(row.answer) - 2, math.floor((len(row.answer) - 1) * fraction)) for row in rows]
            suffixes = [row.answer[length:] for row, length in zip(rows, lengths)]
            chunk = answers([row.prompt + row.answer[:length] for row, length in zip(rows, lengths)],
                            map(len, suffixes), suffixes, device)
            totals, _ = score(chunk, *rollout(model, chunk), final_token=False)
            correct, exact = correct + int(totals[0]), exact + int(totals[1])
            total, provided = total + sum(map(len, suffixes)), provided + sum(lengths)
        results.append({"prefix_fraction": fraction, "provided_payload_tokens": provided,
                        "suffix_exact_accuracy": exact / len(examples), "suffix_token_accuracy": correct / total,
                        "examples": len(examples), "suffix_target_tokens": total,
                        "scope": "true_partial_prefix; at_least_one_unseen_payload_token; greedy_suffix_includes_EOS"})
    return results
