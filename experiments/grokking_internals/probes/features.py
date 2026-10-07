"""Train-only frozen linear probes, centered spectra and FFN neuron profiles.

Sources: Nanda et al., arXiv:2301.05217v1, section 4 (linear directions
and low rank); Golechha, arXiv:2405.12755v1, section 3.1 (sparsity).
Deviations: a fixed ridge probe replaces frequency-selected directions;
spectra use centered activation covariance energy; sparsity includes exact
zeros and train-RMS-relative thresholds. No held-out labels fit a probe or
choose ablations. None of these statistics certifies a correct rule.
"""

import torch
from torch.nn import functional as F


def fit_linear(features, targets, classes, ridge=1e-3):
    """Fit only the supplied training examples, with a fixed ridge strength."""
    x = features.detach().double()
    mean = x.mean(0)
    scale = (x - mean).square().mean(0).sqrt().clamp_min(1e-12)
    z = (x - mean) / scale
    y = F.one_hot(targets, classes).double()
    bias = y.mean(0)
    covariance = z.T @ z / len(z)
    weights = torch.linalg.solve(covariance + ridge * torch.eye(z.shape[1]), z.T @ (y - bias) / len(z))
    return mean, scale, weights, bias


def linear_scores(fit, features):
    mean, scale, weights, bias = fit
    return (features.double() - mean) / scale @ weights + bias


def linear_probe(features, targets, train, heldout, classes):
    fit = fit_linear(features[train], targets[train], classes)
    scores = linear_scores(fit, features)
    shuffled = targets[train][torch.randperm(len(train), generator=torch.Generator().manual_seed(0))]
    null_fit = fit_linear(features[train], shuffled, classes)
    null_predictions = linear_scores(null_fit, features[heldout]).argmax(-1)
    result = {"ridge": 1e-3, "fit_examples": len(train), "test_examples": len(heldout),
              "shuffled_train_labels_heldout_accuracy": float((null_predictions == targets[heldout]).double().mean())}
    for label, selected in (("train", train), ("heldout", heldout)):
        prediction = scores[selected].argmax(-1)
        result[f"{label}_accuracy"] = float((prediction == targets[selected]).double().mean())
    return result


def spectrum(features):
    """Eigenvalue-energy entropy rank; not entropy of singular values themselves."""
    x = features.detach().double()
    mean = x.mean(0)
    x = x - mean
    values, directions = torch.linalg.eigh(x.T @ x / len(x))
    values, directions = values.flip(0).clamp_min(0), directions.flip(1)
    total = float(values.sum())
    if total <= 1e-12:
        result = {"energy_entropy_rank": None, "stable_rank": None, "rank99": None,
                  "top8_energy_fraction": None, "total_variance": total}
    else:
        p = values / total
        entropy = -(p * p.clamp_min(1e-300).log()).sum()
        result = {"energy_entropy_rank": float(entropy.exp()), "stable_rank": total / float(values[0]),
                  "rank99": int(torch.searchsorted(p.cumsum(0), torch.tensor(.99, dtype=p.dtype))) + 1,
                  "top8_energy_fraction": float(p[:8].sum()), "total_variance": total}
    result["eigenvalue_energy_fractions"] = (values / total).tolist() if total > 1e-12 else []
    return result, mean, directions


def profiles(features, labels, classes):
    x = features.detach().double()
    sums = torch.zeros(classes, x.shape[1], dtype=torch.float64)
    sums.index_add_(0, labels, x)
    counts = torch.bincount(labels, minlength=classes)
    means = sums / counts.clamp_min(1)[:, None]
    centered = means - means.mean(0)
    peak = means.max(0).values
    others = (means.sum(0) - peak) / max(1, classes - 1)
    selectivity = (peak - others) / (peak + others).clamp_min(1e-12)
    return centered, selectivity, counts


def neurons(features, labels, train, heldout, classes):
    """Train-selected neuron profiles, excluding the trivial zero quotient."""
    train = train[labels[train] != 0]
    heldout = heldout[labels[heldout] != 0]
    # Nonzero classes are relabeled; every profile requires both-split coverage.
    a, b = features[train].double(), features[heldout].double()
    train_profile, selectivity, tc = profiles(a, labels[train] - 1, classes - 1)
    test_profile, test_selectivity, hc = profiles(b, labels[heldout] - 1, classes - 1)
    coverage = bool((tc >= 2).all() and (hc >= 2).all())
    norm = train_profile.norm(dim=0) * test_profile.norm(dim=0)
    cosine = (train_profile * test_profile).sum(0) / norm.clamp_min(1e-12)
    valid = norm > 1e-12
    ranking = selectivity.argsort(descending=True)
    scale = a.square().mean(0).sqrt()
    result = {"nonzero_quotients_only": True, "class_coverage": coverage,
              "units": a.shape[1], "dead_on_train_fraction": float((scale == 0).double().mean()),
              "exact_zero_train_fraction": float((a == 0).double().mean()),
              "exact_zero_heldout_fraction": float((b == 0).double().mean()),
              "train_rms": float(a.square().mean().sqrt()), "heldout_rms": float(b.square().mean().sqrt()),
              "median_train_selectivity": float(selectivity.median()),
              "median_profile_cosine": float(cosine[valid].median()) if coverage and bool(valid.any()) else None,
              "top_train_selected_units": []}
    for threshold in (.001, .01):
        result[f"relative_zero_{threshold:g}_train_fraction"] = float((a.abs() <= threshold * scale).double().mean())
        result[f"relative_zero_{threshold:g}_heldout_fraction"] = float((b.abs() <= threshold * scale).double().mean())
    for unit in ranking[:10].tolist():
        result["top_train_selected_units"].append({"unit": unit, "train_selectivity": float(selectivity[unit]),
            "heldout_selectivity": float(test_selectivity[unit]) if coverage else None,
            "profile_cosine": float(cosine[unit]) if coverage and bool(valid[unit]) else None,
            "train_active_fraction": float((a[:, unit] > 0).double().mean())})
    return result
