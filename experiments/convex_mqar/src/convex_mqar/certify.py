"""Convex metric training and numerical MQAR recall.

New extension, not a reproduction of either paper's neural training protocol.
Sources (local): 2211.11052v1 §3.1–3.2; 2312.04927v1 app:synthetic.
Uses fixed binary codes, known key/value prefix layout, and one-bit matching
calibration. The control is constructed dot-product softmax attention, not
an independently trained GPT model. No key/value dictionary is memorized.
Run from the project directory: uv run convex-mqar certify
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize


def train_metric(bits, rng):
    """Solve the hinge epigraph QP, with every metric coordinate trainable."""
    initial_w = rng.normal(0, 2, bits)
    initial = np.r_[initial_w, np.maximum(1 - initial_w, 0) + 0.1]
    constraint = LinearConstraint(np.c_[np.eye(bits), np.eye(bits)], 1, np.inf)
    bounds = Bounds(np.r_[np.full(bits, -np.inf), np.zeros(bits)], np.inf)

    def objective(z):
        return z[:bits] @ z[:bits] / 4 + z[bits:].sum()

    def gradient(z):
        return np.r_[z[:bits] / 2, np.ones(bits)]

    result = minimize(objective, initial, jac=gradient, method="SLSQP",
                      bounds=bounds, constraints=constraint,
                      options={"ftol": 1e-12, "maxiter": 500})
    if not result.success:
        raise RuntimeError(result.message)
    raw_w = result.x[:bits]
    if np.max(np.abs(raw_w - 1)) > 1e-6:
        raise AssertionError("The solver did not recover the certified optimum")
    # Enforce the proved lower bound after floating-point optimization.
    w = np.maximum(raw_w, 1.0)
    return w, {"solver": "SLSQP (convex QP)", "iterations": int(result.nit),
               "calibration_pairs": bits, "objective": float(result.fun),
               "certified_global_minimum": bits / 4,
               "raw_metric_max_error": float(np.max(np.abs(raw_w - 1))),
               "lower_bound_repair": float(np.max(w - raw_w))}


def encode(tokens, bits):
    return ((np.asarray(tokens)[..., None] >> np.arange(bits)) & 1).astype(float)


def hamming_cost(queries, keys, w):
    # Ordinary products on fixed features; no token-equality lookup.
    weighted_queries = queries * w
    return ((weighted_queries.sum(-1))[:, None] +
            (keys * w).sum(-1)[None, :] - 2 * weighted_queries @ keys.T)


def simplex_projection(z):
    """Sparsemax: Euclidean projection, which solves ||a||²/4 + cost·a."""
    ordered = np.sort(z, axis=1)[:, ::-1]
    cumulative = np.cumsum(ordered, axis=1)
    ranks = np.arange(1, z.shape[1] + 1)
    count = (1 + ranks * ordered > cumulative).sum(axis=1)
    threshold = (cumulative[np.arange(len(z)), count - 1] - 1) / count
    return np.maximum(z - threshold[:, None], 0)


def value_decode(weights, values, null=None):
    """Mix one-hot values and decode the largest value coordinate."""
    unique_values, groups = np.unique(values, return_inverse=True)
    logits = np.zeros((len(weights), len(unique_values)))
    np.add.at(logits, (np.arange(len(weights))[:, None], groups[None, :]), weights)
    prediction = unique_values[logits.argmax(axis=1)]
    if null is not None:
        prediction[null > logits.max(axis=1)] = -1
    return prediction


def make_example(rng, length, pairs, vocab, alpha):
    """Explicit MQAR variant of the paper's underspecified Procedure 1.

    Distinct keys; independently resampled values; prefix of adjacent pairs;
    one repeat per key at a distinct position sampled with a power law;
    unused positions filled with value tokens. Position weights are p^-alpha.
    """
    half = vocab // 2
    keys = rng.choice(half, size=pairs, replace=False)
    values = rng.integers(half, vocab, size=pairs)
    tokens = rng.integers(half, vocab, size=length)
    tokens[:2 * pairs:2], tokens[1:2 * pairs:2] = keys, values
    available = np.arange(2 * pairs, length)
    probability = available.astype(float) ** (-alpha)
    positions = rng.choice(available, size=pairs, replace=False,
                           p=probability / probability.sum())
    order = rng.permutation(pairs)
    tokens[positions] = keys[order]
    return tokens, positions, values[order]


def evaluate(rng, w, length, cases, vocab, alpha):
    pairs = length // 4
    bits = len(w)
    correct = softmax_correct = first_keys_zero = queries = 0
    largest_weight_error = 0.0
    for _ in range(cases):
        raw, test_positions, labels = make_example(rng, length, pairs, vocab, alpha)
        # The prefix format is known; this adapter supplies local key/value
        # alignment. Routing receives all key-token positions, not the answers.
        keys, values = raw[:2 * pairs:2], raw[1:2 * pairs:2]
        key_positions = np.flatnonzero(raw < vocab // 2)
        cost = hamming_cost(encode(raw[key_positions], bits), encode(keys, bits), w)
        causal = 2 * np.arange(pairs)[None, :] + 1 < key_positions[:, None]
        z = np.c_[-2 * cost, -np.ones(len(key_positions))]
        z[:, :pairs] = np.where(causal, z[:, :pairs], -np.inf)
        weights = simplex_projection(z)
        if not np.allclose(weights.sum(1), 1, atol=1e-10):
            raise AssertionError("Invalid simplex weights")
        if np.any(weights[:, :pairs][~causal] != 0):
            raise AssertionError("A future pair received attention")
        decoded = value_decode(weights[:, :pairs], values, weights[:, -1])
        rows = np.searchsorted(key_positions, test_positions)
        correct += int((decoded[rows] == labels).sum())
        prefix_rows = np.searchsorted(key_positions, 2 * np.arange(pairs))
        first_keys_zero += int((decoded[prefix_rows] == -1).sum())
        largest_weight_error = max(largest_weight_error,
                                   float(np.max(1 - weights.max(axis=1))))
        # Standard softmax of binary-code dot products. For positive w,
        # -cost differs from 1/2 * <sqrt(w)*(2q-1), sqrt(w)*(2k-1)>
        # by a row-constant, so these are genuine dot-product weights.
        beta = np.log(2 * pairs)
        score = -beta * cost[rows]
        score -= score.max(axis=1, keepdims=True)
        control = np.exp(score)
        control /= control.sum(axis=1, keepdims=True)
        softmax_correct += int((value_decode(control, values) == labels).sum())
        queries += pairs
    return {"length": length, "pairs": pairs, "sequences": cases,
            "query_count": queries, "convex_correct": correct,
            "softmax_correct": softmax_correct,
            "convex_accuracy": correct / queries,
            "softmax_accuracy": softmax_correct / queries,
            "first_key_count": queries, "first_keys_abstained": first_keys_zero,
            "largest_weight_error": largest_weight_error}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.cases <= 0:
        parser.error("--cases must be positive")
    vocab, bits, alpha = 8192, 13, 0.1
    training_rng, test_rng = [np.random.default_rng(s)
                              for s in np.random.SeedSequence(args.seed).spawn(2)]
    w, training = train_metric(bits, training_rng)
    report = {"scope": "Recall comparison; not end-to-end GPT training parity",
              "seed": args.seed, "vocabulary": vocab, "metric_bits": bits,
              "alpha": alpha, "metric": w.tolist(), "training": training,
              "results": []}
    for length in (64, 128, 256, 512):
        row = evaluate(test_rng, w, length, args.cases, vocab, alpha)
        report["results"].append(row)
        print(json.dumps(row), flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"training": training}), flush=True)


if __name__ == "__main__":
    main()
