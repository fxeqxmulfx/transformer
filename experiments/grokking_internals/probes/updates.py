"""Separate logit rescaling from changes in decision geometry at endpoints.

Source: Prieto et al., arXiv:2501.04697v1, section on naive loss
minimization and its definition. Deviation: project the measured logit change
between saved endpoints onto previous row-centered logits; do not assume
GPTMini is homogeneous, equate radial weight motion with scaling, or
change AdamW/softmax. A multi-update endpoint projection is not proof of
an NLM parameter direction throughout the interval or on unseen domains.
"""

import torch
from torch.nn import functional as F


def decompose(before, after, targets):
    if before.shape != after.shape or before.ndim != 2 or len(targets) != len(before):
        raise ValueError("Functional endpoints and targets must have matching rows")
    a, b = before.detach().double(), after.detach().double()
    a, b = a - a.mean(-1, keepdim=True), b - b.mean(-1, keepdim=True)
    delta = b - a
    previous, changed = float(a.square().sum()), float(delta.square().sum())
    alpha = float((a * delta).sum()) / previous if previous > 1e-12 else None
    scaled = alpha * a if alpha is not None else torch.zeros_like(a)
    residual = delta - scaled
    scale_energy = float(scaled.square().sum())
    residual_energy = float(residual.square().sum())
    fitted_scale = 1 + alpha if alpha is not None else None
    hypothetical = fitted_scale * a if fitted_scale is not None else a
    return {"fitted_global_scale": fitted_scale,
            "scale_change_energy_fraction": min(1., max(0., scale_energy / changed)) if changed > 1e-12 and alpha is not None else None,
            "geometry_change_energy_fraction": min(1., max(0., residual_energy / changed)) if changed > 1e-12 else None,
            "change_rms": float(delta.square().mean().sqrt()),
            "energy_decomposition_error": abs(changed - scale_energy - residual_energy) / max(1., changed),
            "prediction_change_fraction": float((a.argmax(-1) != b.argmax(-1)).double().mean()),
            "before_loss": float(F.cross_entropy(a, targets)), "after_loss": float(F.cross_entropy(b, targets)),
            "global_scaling_only_loss": float(F.cross_entropy(hypothetical, targets)),
            "global_scaling_preserves_decisions": fitted_scale is not None and fitted_scale > 0,
            "before_accuracy": float((a.argmax(-1) == targets).double().mean()),
            "after_accuracy": float((b.argmax(-1) == targets).double().mean())}


def measure(before, after, targets, train, heldout, earlier_step, step):
    if earlier_step >= step:
        raise ValueError("Functional changes require increasing checkpoint updates")
    return {"from_step": earlier_step, "to_step": step, "interval_updates": step - earlier_step,
            "scope": "endpoint_functional_projection; not_a_single_optimizer_update_or_parameter_homogeneity_claim",
            "train": decompose(before[train], after[train], targets[train]),
            "heldout": decompose(before[heldout], after[heldout], targets[heldout])}
