"""What a study's splits hold: repeated and shared inputs, and conflicting labels of one training context.

A port of `corpus_report` and `context_report` of
`experiments/synthetic_trainers/corpus.py`. An input is what its generator
identifies a problem by (`Generator.key`): the prompt, the whole input of a
prefix task, or the whole payload of the random control.
"""

from collections import Counter
import math

from .vocabulary import IGNORE


def corpus_report(splits):
    """Each split's rows, distinct inputs and repeated rows, and the inputs each pair of splits shares."""
    keys = {name: Counter(map(split.generator.key, split.examples)) for name, split in splits.items()}
    sizes = {name: {"examples": sum(counts.values()), "unique_inputs": len(counts),
                    "duplicate_rows": sum(counts.values()) - len(counts)} for name, counts in keys.items()}
    overlaps = {}
    names = tuple(keys)
    for index, first in enumerate(names):
        for second in names[index + 1:]:
            shared = keys[first].keys() & keys[second].keys()
            overlaps[f"{first}/{second}"] = {"unique_inputs": len(shared),
                                             "rows_in_first": sum(keys[first][key] for key in shared),
                                             "rows_in_second": sum(keys[second][key] for key in shared)}
    return {"splits": sizes, "overlaps": overlaps, "identity": "raw_prompt_or_prefix_input; random_lm_full_payload"}


def context_report(examples):
    """The least error and loss any deterministic causal predictor can reach on these supervised contexts.

    A context is a supervised position's input prefix; labels that disagree on
    one context bound every predictor that reads only it. An empirical floor
    on these rows, not a population risk.
    """
    contexts = {}
    for example in examples:
        for position, target in enumerate(example.targets):
            if target != IGNORE:
                contexts.setdefault(example.tokens[:position + 1], Counter())[target] += 1
    total = sum(sum(labels.values()) for labels in contexts.values())
    loss = error = conflicting = 0
    for labels in contexts.values():
        count = sum(labels.values())
        conflicting += len(labels) > 1
        error += count - max(labels.values())
        loss += sum(-n * math.log(n / count) for n in labels.values())
    return {"supervised_contexts": len(contexts), "conflicting_contexts": conflicting,
            "empirical_min_token_error": error / total, "empirical_min_token_loss_nats": loss / total,
            "scope": "any deterministic causal predictor on these observed contexts"}
