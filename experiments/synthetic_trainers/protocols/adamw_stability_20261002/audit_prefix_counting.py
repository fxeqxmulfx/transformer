"""Audit the prepared C-RASP program against independent paper point semantics.

Knee-Deep in C-RASP, Section 2.3: inclusive past counts and derived integer
operations. This checks an existing task oracle; it is not a training campaign.
"""

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from itertools import product
import json
from pathlib import Path
import random
import sys

from experiments.synthetic_trainers import vocabulary as v
from experiments.synthetic_trainers.crasp import evaluate_formula, program_with_witnesses
from experiments.synthetic_trainers.tests.test_prefix_programs import point_semantics


def audit():
    formula, _ = program_with_witnesses(2, 0)
    alphabet = (v.A, v.B, v.C)
    bank, order_witness = {}, None
    counts = {"short_words": 0, "short_prefix_labels": 0, "long_words": 0, "long_prefix_labels": 0}

    def check(word, category):
        actual = evaluate_formula(formula, word)
        expected = tuple(point_semantics(formula, word, i) for i in range(len(word)))
        if actual != expected:
            raise ValueError("The prepared counting oracle differs from independent point semantics")
        counts[category + "_words"] += 1
        counts[category + "_prefix_labels"] += len(word)
        return actual

    for word in product(alphabet, repeat=7):
        labels = check(word, "short")
        histogram = tuple(Counter(word)[token] for token in alphabet)
        if min(histogram) and order_witness is None:
            previous = bank.setdefault(histogram, (word, labels))
            if previous[1][-1] != labels[-1]:
                order_witness = {"word_length": 7, "input_length_with_BOS": 8,
                    "histogram": histogram,
                    "word_a": [v.token_name(token) for token in previous[0]],
                    "word_b": [v.token_name(token) for token in word],
                    "labels_a": previous[1], "labels_b": labels}
    rng = random.Random(713)
    for _ in range(128):
        check(tuple(rng.choice(alphabet) for _ in range(15)), "long")
    if not order_witness or "torch" in sys.modules:
        raise ValueError("Need an order-sensitive witness and verification without Torch")
    sources = ("experiments/synthetic_trainers/crasp.py", "experiments/synthetic_trainers/vocabulary.py",
               "experiments/synthetic_trainers/tests/test_prefix_programs.py",
               "papers/arXiv-2506.16055v3/neurips2025.tex")
    return {"recorded_at_utc": datetime.now(timezone.utc).isoformat(), "verified": True,
        "program": {"count_depth": formula.depth, "formula_seed": 0, "ast": asdict(formula),
                    "expression": formula.describe()}, "checked_support": counts,
        "short_support": "all_3_to_the_7_words; all_inclusive_prefix_labels",
        "long_support": "128_diagnostic_words_of_length_15; seed_713; not_training_or_confirmation_seeds",
        "same_histogram_order_witness": order_witness, "torch_imported": False,
        "source_hashes": {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in sources},
        "auditor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "paper_scope": "Section_2.3_past_counting_semantics; integer_negation_and_constants_are_derived_operations",
        "scope": "finite_oracle_and_order_identifiability_checks; no_learning_depth_lower_bound_or_transfer_claim"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit()
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"verified": True, "checked_support": result["checked_support"],
                      "order_witness": result["same_histogram_order_witness"], "torch_imported": False}))
