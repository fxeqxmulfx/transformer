"""Causal support statistics for EXPERIMENT_PLAN.md, step 2.

The strict positive support and the unit score gap are those of
arXiv:1602.02068v2, section 2.2. These are floating-point observations, not
certificates of derivatives. Future and padded positions never enter a
denominator. Row means and the share of zeros among visible pairs have
different denominators and are recorded separately. Normalized Shannon
entropy is zero for a row with one visible position. Turnover compares the
same causal (example, query, key) pairs at consecutive observations.
"""

import torch


@torch.no_grad()
def statistics(scores, weights, lengths, supervised, previous=None):
    """Statistics per head for all rows, non-BOS rows and supervised rows, plus the support on CPU."""
    if scores.ndim != 4 or scores.shape != weights.shape or scores.shape[-2] != scores.shape[-1]:
        raise ValueError("Attention statistics need equal (batch, heads, length, length) scores and weights")
    batch, heads, length, _ = scores.shape
    if lengths.shape != (batch,) or supervised.shape != (batch, length):
        raise ValueError("Lengths and supervised rows must describe the attention batch")
    if length < 1 or not bool(((lengths >= 1) & (lengths <= length)).all()):
        raise ValueError("Every example has between one and length context positions")
    if previous is not None and previous.shape != weights.shape:
        raise ValueError("Previous supports must have the same fixed example and attention shape")
    positions = torch.arange(length, device=scores.device)
    valid = positions[None, :] < lengths[:, None]
    visible = valid[:, None, :, None] & (positions[:, None] >= positions[None, :])[None, None]
    support = (weights > 0) & visible
    count = support.sum(-1)
    self_only = (count == 1) & support.diagonal(dim1=-2, dim2=-1)
    prefix = (positions + 1).to(scores.dtype)
    values = weights.masked_fill(~visible, 0)
    entropy = -torch.special.xlogy(values, values).sum(-1) / prefix.log().clamp_min(1e-12)
    masked_scores = scores.masked_fill(~visible, -torch.inf)
    if length == 1:
        gaps = torch.full_like(count, torch.inf, dtype=scores.dtype)
    else:
        top = masked_scores.topk(2, dim=-1).values
        gaps = top[..., 0] - top[..., 1]
    finite_scores = scores.masked_fill(~visible, 0)
    mean = finite_scores.sum(-1) / prefix
    deviations = (scores - mean[..., None]).masked_fill(~visible, 0)
    variance = deviations.square().sum(-1) / prefix
    changed = None if previous is None else (support != previous.to(support.device)) & visible

    def summarize(rows):
        size, pairs = int(rows.sum()), int((rows * (positions + 1)).sum())
        if not size:
            return [{"head": head, "row_count": 0, "visible_pair_count": 0} for head in range(heads)]
        chosen = rows[:, None]
        totals = {
            "mean_support_positions": (count * chosen).sum((0, 2)) / size,
            "mean_support_share": (count / prefix * chosen).sum((0, 2)) / size,
            "exact_zero_pair_share": 1 - (count * chosen).sum((0, 2)) / pairs,
            "singleton_row_share": ((count == 1) & chosen).sum((0, 2)) / size,
            "self_only_row_share": (self_only & chosen).sum((0, 2)) / size,
            "score_gap_gt_one_share": ((gaps > 1) & chosen).sum((0, 2)) / size,
            "normalized_entropy": (entropy * chosen).sum((0, 2)) / size,
            "mean_score_std": (variance.sqrt() * chosen).sum((0, 2)) / size,
        }
        changed_counts = None
        if changed is not None:
            changed_totals = (changed & chosen[..., None]).sum((0, 2, 3))
            totals["turnover_share"] = changed_totals / pairs
            changed_counts = changed_totals.cpu().tolist()
        columns = torch.stack(list(totals.values()), dim=1).cpu().tolist()
        result = []
        for head, column in enumerate(columns):
            row = {"head": head, "row_count": size, "visible_pair_count": pairs, **dict(zip(totals, column))}
            row["turnover_changed_pairs"] = None if changed_counts is None else changed_counts[head]
            row.setdefault("turnover_share", None)
            result.append(row)
        return result

    return {"all_rows": summarize(valid), "nontrivial_rows": summarize(valid & (positions > 0)),
            "supervised_rows": summarize(valid & supervised)}, support.cpu()
