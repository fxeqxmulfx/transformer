"""Evaluate the proved cleanup inequality on current machine logits.

Source: Transformer.Grokking.Geometry.Decisions and Cleanup at 3661a44;
Nanda et al., arXiv:2301.05217v1, section 5.1. Deviation: fixed division
cells replace final-selected frequencies. Correct margins use raw cell
means, retaining class bias. Errors remove only harmless row offsets.

Certificate counts use exact integers/Fractions representing the observed
finite binary floats. Quantiles and energy summaries use float64. Neither
the Python program nor floating-point transformer forward is proved in
Lean; the exact sufficient inequality is. No future success is predicted.
"""

from fractions import Fraction
import math

import torch


def exact_cell(rows, target):
    """Apply `2 * R < margin^2` exactly to one nonempty observed cell.

    Source: Geometry.strictCorrect_of_margin_and_energy at 3661a44.
    For n rows, C classes and common binary denominator D, the aligned
    residual numerator is n*C*x - n*row_sum - C*column_sum + total_sum.
    Removing the row offset preserves the original strict argmax.
    """
    n, classes = len(rows), len(rows[0]) if rows else 0
    if n == 0 or classes < 2 or not 0 <= target < classes:
        raise ValueError("A cell needs observations, at least two classes and a valid target")
    if any(len(row) != classes or any(not math.isfinite(x) for x in row) for row in rows):
        raise ValueError("Cell logits must form a finite rectangular array")
    pairs = [[float(x).as_integer_ratio() for x in row] for row in rows]
    denominator = max(d for row in pairs for _, d in row)
    integers = [[a * (denominator // d) for a, d in row] for row in pairs]
    columns = [sum(row[k] for row in integers) for k in range(classes)]
    total = sum(columns)
    gap = columns[target] - max(columns[k] for k in range(classes) if k != target)
    squared = []
    strict = []
    for row in integers:
        row_sum = sum(row)
        residual = [n * classes * x - n * row_sum - classes * column + total
                    for x, column in zip(row, columns)]
        squared.append(sum(x * x for x in residual))
        strict.append(row[target] > max(row[k] for k in range(classes) if k != target))
    energies = [Fraction(value, (n * classes * denominator) ** 2) for value in squared]
    return {"margin": Fraction(gap, n * denominator), "point_energies": energies,
            "residual_sum": sum(energies, Fraction(0)), "strict_correct": strict,
            "certified": [gap > 0 and 2 * value < (classes * gap) ** 2 for value in squared]}


def _quantiles(values):
    if values.numel() == 0:
        return None
    values = values.detach().double().flatten()
    if not torch.isfinite(values).all():
        raise FloatingPointError("Diagnostic summaries require finite float64 values")
    levels = torch.tensor([.05, .5, .95], dtype=values.dtype, device=values.device)
    return dict(zip(("p05", "median", "p95"), map(float, torch.quantile(values, levels))))


@torch.no_grad()
def measure(logits, selected, targets, energy_floor=1e-12):
    """Held-out-only cell means and exact current-logit certificate counts.

    Source: Geometry.cell_cleanup_certifies at 3661a44; the selected-cell
    convention of lab.infrastructure.engine.grokking at 4436290. Cell zero
    is excluded from structural measurements and retained in raw accuracy.
    Targets must be constant within a cell. Numerical signed safety is
    margin / sqrt(margin^2 + 2*point_error), with zero denominator absent.
    It is scale independent, label dependent and not a future guarantee.
    """
    if logits.ndim != 3 or logits.shape[0] < 2 or logits.shape[1] == 0 or logits.shape[2] < 2:
        raise ValueError("Logits need cell x observation x class dimensions")
    if not logits.is_floating_point() or not torch.isfinite(logits).all():
        raise ValueError("Logits must be finite floating-point values")
    if not math.isfinite(energy_floor) or energy_floor <= 0:
        raise ValueError("Energy floor must be positive and finite")
    qcount, orbit, classes = logits.shape
    if selected.ndim != 1 or selected.dtype != torch.long or targets.dtype != torch.long:
        raise ValueError("Selected indices and targets must be integer tensors")
    if targets.numel() != qcount * orbit or selected.numel() == 0:
        raise ValueError("Targets must cover every input and the selection must be nonempty")
    selected = selected.to(logits.device)
    targets = targets.to(logits.device).reshape(qcount, orbit)
    if selected.min() < 0 or selected.max() >= qcount * orbit or selected.unique().numel() != selected.numel():
        raise ValueError("Selected indices must be in range and unique")
    if targets.min() < 0 or targets.max() >= classes or not (targets == targets[:, :1]).all():
        raise ValueError("Each cell must have one valid constant target")
    x = logits.detach().double()
    mask = torch.zeros((qcount, orbit), dtype=torch.bool, device=x.device)
    mask.reshape(-1)[selected] = True
    raw_accuracy = float((x.argmax(-1)[mask] == targets[mask]).double().mean())
    mask[0] = False
    if not mask.any():
        raise ValueError("Structural observation needs a selected nonzero cell")
    counts = mask.sum(1)
    rows = x - x.mean(-1, keepdim=True)
    bias = rows[mask].mean(0)
    z = rows - bias
    cell = (z * mask[..., None]).sum(1) / counts.clamp_min(1)[:, None]
    reference = cell + bias
    labels = targets[:, 0]
    competitors = reference.clone()
    competitors.scatter_(1, labels[:, None], -torch.inf)
    margins = reference.gather(1, labels[:, None]).squeeze(1) - competitors.max(1).values
    point_error = (z - cell[:, None, :]).square().sum(-1)
    point_total = z.square().sum(-1)
    projected = cell.square().sum(-1)[:, None].expand_as(point_error)
    total, kept, residual = (float(a[mask].sum()) for a in (point_total, projected, point_error))
    if not all(math.isfinite(a) for a in (total, kept, residual)):
        raise FloatingPointError("Diagnostic energy summaries overflow float64")
    scalar_count = int(mask.sum()) * classes
    energy = {"scalar_coordinates": scalar_count, "total_sum": total, "projected_sum": kept,
              "residual_sum": residual, "total_mean": total / scalar_count,
              "projected_mean": kept / scalar_count, "residual_mean": residual / scalar_count,
              "decomposition_absolute_error": abs(total - kept - residual)}
    fraction = kept / total if energy["total_mean"] > energy_floor else None
    magnitude = margins[:, None].square() + 2 * point_error
    usable = mask & (magnitude > 0)
    safety = margins[:, None].expand_as(point_error)[usable] / magnitude[usable].sqrt()
    numerical = (margins[:, None] > 0) & (2 * point_error < margins[:, None].square())
    exact, exact_strict, cell_records = [], [], []
    exact_numerical_disagreements = 0
    for q in range(1, qcount):
        members = mask[q].nonzero().flatten()
        if members.numel() == 0:
            continue
        result = exact_cell(x[q, members].tolist(), int(labels[q]))
        exact.extend(result["certified"])
        exact_strict.extend(result["strict_correct"])
        exact_numerical_disagreements += sum(a != b for a, b in
            zip(result["certified"], numerical[q, members].tolist()))
        cell_records.append(result)
    global_residual = sum((r["residual_sum"] for r in cell_records), Fraction(0))
    global_count = sum(len(r["certified"]) for r in cell_records
                       if r["margin"] > 0 and 2 * global_residual < r["margin"] ** 2)
    certified_count = sum(exact)
    violations = sum(certificate and not correct for certificate, correct in zip(exact, exact_strict))
    selected_count = len(exact)
    positive_count = sum(len(r["certified"]) for r in cell_records if r["margin"] > 0)
    valid_cells = counts[1:] >= 2
    actual = (x.argmax(-1) == targets)[mask]
    return {"selected_nonzero_examples": selected_count, "selected_nonzero_cells": len(cell_records),
            "raw_selected_accuracy_including_zero": raw_accuracy,
            "raw_nonzero_accuracy": float(actual.double().mean()),
            "raw_nonzero_strict_correct_fraction": sum(exact_strict) / selected_count,
            "reference_nonzero_accuracy": float((reference.argmax(1)[:, None].expand_as(targets) == targets)[mask].double().mean()),
            "positive_reference_margin_fraction": positive_count / selected_count,
            "certified_count": certified_count, "certified_fraction": certified_count / selected_count,
            "globally_certified_count": global_count, "globally_certified_fraction": global_count / selected_count,
            "certified_incorrect_count": violations,
            "exact_float64_certificate_disagreements": exact_numerical_disagreements,
            "minimum_cell_size": int(counts[1:].min()), "two_point_cell_coverage": float(valid_cells.double().mean()),
            "structural_fraction": fraction,
            "structure_protocol_eligible": bool(valid_cells.all()) and fraction is not None,
            "energy": energy, "reference_margin": _quantiles(margins[:, None].expand_as(point_error)[mask]),
            "point_residual_energy": _quantiles(point_error[mask]), "signed_safety": _quantiles(safety),
            "mean_signed_safety": float(safety.mean()) if safety.numel() else None,
            "undefined_signed_safety_count": selected_count - int(usable.sum()),
            "scope": "exact_sufficient_inequality_on_current_machine_logits; float64_summaries; no_future_guarantee"}
