"""Check character separation, not just rank, against complete checkpoint predictions.

Source: review/flat_spectrum.md, Powered division and the current Rotate90 model.
Exact algebra is checked by exhaustive finite arithmetic; approximate spectral
concentration is only the prospective hypothesis in hypotheses.json.
Run from python/: uv run --locked python ../experiments/rotate90_power_learning/theory.py
"""

import hashlib
import json
import math
from pathlib import Path

import torch

from lab.dsl import PowerDivision
from lab.infrastructure.benchmarks import build_task
from lab.infrastructure.loader import load
from lab.infrastructure.nn import build_model
from lab.infrastructure.optim import build_optimizer, evaluation_weights
from lab.infrastructure.rotary_geometry import multiplicative_ordering
from lab.infrastructure.store import RunDirectory

ROOT = Path(__file__).resolve().parent


def rank_one_recovery_pattern(prime, seed):
    """Exact integer graph/matching witnesses for the existing Theorem E criterion.

    Unit-modulus character kernels on nonzero operands are recovered by ideal
    nuclear-norm completion iff the observed bipartite graph is connected and
    contains a perfect matching. This supplies no training convergence claim.
    """
    task = build_task(PowerDivision(prime, 1), seed, torch.device("cpu"))
    n = prime - 1
    neighbors = [set() for _ in range(n)]
    adjacency = [set() for _ in range(2 * n)]
    for a, b, _ in task.splits["train"].tolist():
        if a:
            neighbors[a - 1].add(b - 1)
            adjacency[a - 1].add(n + b - 1)
            adjacency[n + b - 1].add(a - 1)
    reached, pending = {0}, [0]
    while pending:
        vertex = pending.pop()
        for other in adjacency[vertex] - reached:
            reached.add(other)
            pending.append(other)
    matching = [-1] * n

    def augment(a, visited):
        for b in sorted(neighbors[a]):
            if b not in visited:
                visited.add(b)
                if matching[b] == -1 or augment(matching[b], visited):
                    matching[b] = a
                    return True
        return False

    perfect = all(augment(a, set()) for a in range(n))
    if perfect:
        assert sorted(matching) == list(range(n))
        assert all(b in neighbors[a] for b, a in enumerate(matching))
    connected = len(reached) == 2 * n
    return {"prime": prime, "data_seed": seed, "nonzero_training_cells": sum(map(len, neighbors)),
            "connected": connected, "perfect_matching": perfect,
            "column_to_row_witness_zero_indexed": matching if perfect else None,
            "ideal_character_kernel_uniquely_recovered": connected and perfect}


def small_energy_counterexample():
    """Four centered labels: parity has 99% energy, tiny faithful features decode all."""
    circle = torch.tensor([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]], dtype=torch.float64)
    parity = torch.tensor([10., -10., 10., -10.], dtype=torch.float64)
    embedding = torch.cat((parity[:, None], torch.zeros(4, 1), circle), dim=1)
    states = torch.cat((torch.zeros(4, 2), circle), dim=1)
    scores = states @ embedding.T
    accuracy = float((scores.argmax(1) == torch.arange(4)).double().mean())
    assert accuracy == 1. and bool((embedding.mean(0) == 0).all())
    return {"parity_energy_fraction": 100 / 101, "parity_alias_size": 2,
            "full_readout_accuracy": accuracy, "faithful_energy_fraction": 1 / 101,
            "scope": "A normalized tied-readout counterexample, not a construction of the full network's hidden states."}


def exact_checks(prime, exponent):
    """Compare direct task matrices and analytic character products in complex128."""
    generator, order = multiplicative_ordering(prime)
    n = prime - 1
    d, m = math.gcd(exponent, n), n // math.gcd(exponent, n)
    logs = {value: index for index, value in enumerate(order)}
    labels = torch.tensor([[pow(a * pow(b, -1, prime) % prime, exponent, prime)
                            for b in order] for a in order])
    exponents = torch.tensor([[logs[value] for value in row] for row in labels.tolist()])
    worst = 0.
    for frequency in range(m):
        direct = torch.exp(2j * math.pi * frequency * exponents.double() / n)
        vector = torch.exp(2j * math.pi * frequency * exponent * torch.arange(n).double() / n)
        worst = max(worst, float((direct - vector[:, None] * vector.conj()[None, :]).abs().max()))
    if worst > 1e-10:
        raise AssertionError((prime, exponent, worst))
    fibers = []
    # Integer phases modulo m avoid rounding and test joint, not single, characters.
    for frequencies in ((0,), (1,), (2,), (3,), (6,), (8,), (2, 3), (6, 8)):
        groups = {}
        for value in range(m):
            groups.setdefault(tuple(r * value % m for r in frequencies), []).append(value)
        g = math.gcd(m, *frequencies)
        assert len(groups) == m // g and {len(group) for group in groups.values()} == {g}
        fibers.append({"frequencies": frequencies, "gcd": g, "distinct_codes": len(groups)})
    return {"prime": prime, "exponent": exponent, "generator": generator,
            "image_order": m, "kernel_size": d, "character_factorization_max_error": worst,
            "fiber_checks": fibers}


def output_spectrum(embedding, prime, exponent):
    """Use reachable labels, excluding zero and centering only over that image."""
    generator, _ = multiplicative_ordering(prime)
    d = math.gcd(exponent, prime - 1)
    m = (prime - 1) // d
    order = torch.tensor([pow(generator, d * t, prime) for t in range(m)])
    rows = embedding.detach().cpu().double()[order]
    centered = rows - rows.mean(0)
    energy = torch.fft.fft(centered, dim=0).abs().square().sum(1)
    energy[0] = 0.
    energy /= energy.sum()
    candidates = {g: float(energy[torch.arange(m) % g == 0].sum())
                  for g in range(2, m + 1) if m % g == 0}
    estimates = {str(threshold): max([1] + [g for g, mass in candidates.items() if mass >= threshold])
                 for threshold in (.90, .95, .99, .999)}
    distances = torch.cdist(rows, rows).square()
    distances.fill_diagonal_(float("inf"))
    relative_separation = float(distances.min() / centered.square().sum(1).mean())
    return {"image_order": m, "frequency_energy": energy.tolist(),
            "divisor_energy": candidates, "alias_at_threshold": estimates,
            "minimum_squared_separation_over_mean_centered_energy": relative_separation}


def alias_margin_bound(logits, rows, embedding, prime, g):
    """Validate the current model's final-RMSNorm Cauchy--Schwarz bound.

    For a nontrivial g-th root z, compare the true label c with c*z. The
    normalized state has norm at most sqrt(width), so the gap is at most
    sqrt(width)*||E[c]-E[c*z]||. No energy-to-accuracy assumption is used.
    """
    if g == 1:
        return None
    generator, _ = multiplicative_ordering(prime)
    labels = rows[rows[:, 2] != 0, 2]
    scores = logits[rows[:, 2] != 0].double()
    other = labels * pow(generator, (prime - 1) // g, prime) % prime
    index = torch.arange(len(labels))
    gaps = scores[index, labels] - scores[index, other]
    difference = embedding.detach().cpu().double()[labels] - embedding.detach().cpu().double()[other]
    bound = math.sqrt(embedding.shape[1]) * difference.norm(dim=1)
    excess = float((gaps.abs() - bound).max())
    if excess > 1e-4:
        raise AssertionError(("RMSNorm readout bound", excess))
    return {"pairwise_bound_verified": True, "maximum_excess": excess,
            "maximum_gap_to_bound_ratio": float((gaps.abs() / bound.clamp_min(1e-30)).max()),
            "median_correct_minus_alias_logit": float(gaps.median()),
            "fraction_correct_beats_one_alias": float((gaps > 0).double().mean()),
            "median_pairwise_bound": float(bound.median())}


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    findings = []
    for label, experiment in load(ROOT).experiments.items():
        run = RunDirectory(ROOT / "runs" / label)
        if run.result() is None or experiment.benchmark.train_fraction == 1 or label == "timing":
            continue
        checkpoint = run.checkpoint("cpu")
        task = build_task(experiment.benchmark, experiment.seeds.data, torch.device("cpu"))
        model = build_model(experiment.model, task.vocab, experiment.seeds.model)
        model.load_state_dict(checkpoint["model"])
        optimizer = build_optimizer(experiment.optimizer, model)
        optimizer.load_state_dict(checkpoint["optimizer"])
        with evaluation_weights(optimizer):
            model.eval()
            spectrum = output_spectrum(model.E, task.vocab, experiment.benchmark.exponent)
            rows = task.splits["heldout"]
            logits = torch.cat([model(chunk[:, :2]) for chunk in rows.split(1024)])
            predictions = logits.argmax(-1)
            bound = alias_margin_bound(logits, rows, model.E, task.vocab,
                                       spectrum["alias_at_threshold"]["0.95"])
        g = spectrum["alias_at_threshold"]["0.95"]
        nonzero = rows[:, 2] != 0
        valid = torch.tensor([pow(value, experiment.benchmark.exponent, task.vocab)
                              for value in range(task.vocab)]).unique()
        mapping = torch.tensor([pow(value, g, task.vocab) for value in range(task.vocab)])
        exact = float((predictions[nonzero] == rows[nonzero, 2]).double().mean())
        quotient = float((mapping[predictions[nonzero]] == mapping[rows[nonzero, 2]]).double().mean())
        row = {"label": label, "prime": task.vocab, "exponent": experiment.benchmark.exponent,
               "seed": experiment.seeds.model, "optimizer": type(experiment.optimizer).__name__,
               "spectrum": spectrum, "predicted_alias": g, "nonzero_heldout_accuracy": exact,
               "actual_model_margin_bound": bound,
               "nonzero_heldout_quotient_accuracy": quotient,
               "fraction_predictions_in_valid_image": float(torch.isin(predictions, valid).double().mean()),
               "hypothesis_applies": g > 1,
               "hypothesis_pass": None if g == 1 else quotient > .95 and abs(exact - 1 / g) < .1}
        findings.append(row)
        if label.startswith(("confirm-", "sf-")) or g > 1:
            print(label, "g=", g, "exact=", round(exact, 4), "quotient=", round(quotient, 4),
                  "prediction=", row["hypothesis_pass"], flush=True)
    payload = {"hypotheses_sha256": hashlib.sha256((ROOT / "hypotheses.json").read_bytes()).hexdigest(),
               "large_model_hypotheses_sha256": hashlib.sha256(
                   (ROOT / "large_model_hypotheses.json").read_bytes()).hexdigest(),
               "small_energy_counterexample": small_energy_counterexample(),
               "rank_one_recovery_patterns": [rank_one_recovery_pattern(p, seed)
                                               for p in (17, 97, 193) for seed in (11, 17, 23)],
               "exact_checks": [exact_checks(p, k) for p in (17, 97, 193) for k in (1, 2, 3, 5)],
               "scope": "The algebra is exact; concentration-to-confusion is a diagnostic hypothesis, not an optimization theorem.",
               "runs": findings}
    (ROOT / "theory_results.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
