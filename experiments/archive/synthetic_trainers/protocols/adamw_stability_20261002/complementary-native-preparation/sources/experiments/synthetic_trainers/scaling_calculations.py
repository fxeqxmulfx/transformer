"""Resource arithmetic and a specified finite diversity bound, not an EMC formula."""

import math


def parameter_count(vocab_size, width, layers, heads, ff_multiplier=4):
    """Count the reference model's tied embedding, QKV/projection, FFN, temperatures."""
    if min(vocab_size, width, layers, heads, ff_multiplier) < 1 or width % heads or (width // heads) % 2:
        raise ValueError("Invalid reference-model dimensions")
    return vocab_size * width + layers * ((4 + 2 * ff_multiplier) * width ** 2 + heads)


def motif_bound(max_length=32, alphabet=2, motif_length=4, failure_probability=.05):
    """Cover every (length, start position, motif) under uniform lengths/letters.

    Only the first row of each generator pair is counted, because the rows
    share a length. Its specified event has probability >= 1/(L*A^m).
    A union bound gives P(any event missing) <= M*(1-q)^(N/2).
    This finite coverage property is weaker than the RASP-L Diversity condition
    (arXiv:2310.16028v1, Section 3); it is not a sample-complexity guarantee for
    training, generalization, or double descent.
    """
    if not 1 <= motif_length <= max_length or alphabet < 2 or not 0 < failure_probability < 1:
        raise ValueError("Invalid finite-diversity controls")
    events = alphabet ** motif_length * sum(length - motif_length + 1 for length in range(motif_length, max_length + 1))
    probability = 1 / (max_length * alphabet ** motif_length)
    groups = math.ceil(math.log(events / failure_probability) / -math.log1p(-probability))
    rows = 2 * groups
    rounded = 1 << (rows - 1).bit_length()
    return {"max_length": max_length, "alphabet": alphabet, "motif_length": motif_length,
            "events": events, "minimum_event_probability": probability,
            "failure_probability": failure_probability, "required_rows": rows, "rounded_rows": rounded,
            "rounded_union_failure_bound": events * math.exp((rounded // 2) * math.log1p(-probability)),
            "scope": "specified_finite_motif_coverage; not_RASP_diversity_or_EMC"}


def motif_coverage(split, motif_length=4):
    """Measure the specified coverage on actual oracle-validated binary copy inputs."""
    spec = split.spec
    if spec.task != "copy" or spec.unique or spec.symbols != 2 or spec.min_length != 1:
        raise ValueError("Coverage measurement requires binary copy with lengths 1..L")
    observed = set()
    for example in split.examples:
        word = example.prompt[1:-1]
        for position in range(len(word) - motif_length + 1):
            observed.add((len(word), position, word[position:position + motif_length]))
    expected = motif_bound(spec.length, spec.symbols, motif_length)["events"]
    return {"events": expected, "observed_events": len(observed), "missing_events": expected - len(observed),
            "rows": len(split.examples), "fingerprint": split.fingerprint}


def critical_sample_choice(rows):
    """Choose the next grid's N from train fit only; never inspect test losses.

    A single nested pool supplies a finite frontier, not the expected iid risk
    in DoubleDescent Section 2 Definition 1. Feasible sizes need not be monotone.
    """
    fitted = sorted(row["train_examples"] for row in rows if row["final_fitted"])
    unfitted = sorted(row["train_examples"] for row in rows if not row["final_fitted"])
    higher = [size for size in unfitted if fitted and size > max(fitted)]
    lower, upper = (max(fitted), min(higher)) if higher else (None, None)
    size = (round(math.sqrt(lower * upper)) if lower else
            max(fitted) if fitted else min(unfitted))
    return {"chosen_sample_size": size, "largest_observed_fitted_pool": max(fitted) if fitted else None,
            "first_observed_unfitted_above": min(higher) if higher else None,
            "fitted_sample_sizes": fitted, "unfitted_sample_sizes": unfitted,
            "nonmonotone_fit": bool(fitted and any(n < max(fitted) for n in unfitted)),
            "rule": "geometric_midpoint_of_observed_bracket; otherwise_largest_fit_or_smallest_unfit",
            "scope": "finite_train_fit_calibration; not_population_EMC"}
