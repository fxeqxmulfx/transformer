"""Gradient agreement under the unchanged answer-plus-EOS training objective.

Source objective: Power et al.'s openai/grok, commit
3d64b1d8c1d595dd8ebdb7771998823f1b14c7b3, via ModularTask.loss at 43d4d66.
Deviation: read gradients on eight fixed disjoint batches per split using
autograd.grad; never step an optimizer. Held-out gradients are diagnostic
only. Agreement is empirical, not a convexity or generalization theorem.
"""

import torch

from .capture import evaluating


def group(name):
    if name.startswith("embed."):
        return "embedding"
    if ".attention." in name:
        return "attention"
    if ".ffn." in name:
        return "ffn"
    return "normalization_and_readout"


def cosine(a, b):
    denominator = float(a.norm() * b.norm())
    return max(-1., min(1., float(a @ b) / denominator)) if denominator > 0 else None


def agreement(rows):
    rows = rows.double()
    norms = rows.norm(dim=1)
    rows = rows[norms > 0]
    norms = norms[norms > 0]
    if len(rows) < 2:
        return {"nonzero_batches": len(rows), "mean_pair_cosine": None,
                "positive_pair_fraction": None, "mean_batch_gradient_l2": float(norms.mean()) if len(rows) else 0.}
    units = rows / norms[:, None]
    pairs = (units @ units.T)[torch.triu(torch.ones(len(rows), len(rows), dtype=torch.bool), diagonal=1)]
    return {"nonzero_batches": len(rows), "mean_pair_cosine": float(pairs.mean()),
            "positive_pair_fraction": float((pairs > 0).double().mean()),
            "mean_batch_gradient_l2": float(norms.mean())}


def measure(model, task, batches=8, batch=32):
    """Parameters appear once, including the shared embedding/readout matrix."""
    parameters = list(model.named_parameters())
    device = next(model.parameters()).device
    collected, losses = {}, {}
    with evaluating(model), torch.enable_grad():
        for split in ("train", "heldout"):
            data = task.splits[split]
            count = min(batches * batch, len(data))
            selected = torch.randperm(len(data), generator=torch.Generator().manual_seed(1729))[:count]
            collected[split], losses[split] = [], []
            for indices in selected.split(batch):
                output, targets = task.forward(model, data[indices].to(device))
                loss = task.loss(output, targets)
                gradients = torch.autograd.grad(loss, [p for name, p in parameters], allow_unused=True)
                collected[split].append([torch.zeros_like(p).cpu().flatten() if g is None else g.detach().cpu().flatten()
                                         for (name, p), g in zip(parameters, gradients)])
                losses[split].append(float(loss.detach()))
    groups = {name: [i for i, (key, p) in enumerate(parameters) if group(key) == name]
              for name in sorted({group(key) for key, p in parameters})}
    groups["all"] = list(range(len(parameters)))
    result = {"objective": "mean_answer_and_EOS_cross_entropy; original_training_supervision",
              "batch_size": batch, "batches_by_split": {key: len(rows) for key, rows in collected.items()},
              "sample_seed": 1729, "loss_by_split": {key: sum(values) / len(values) for key, values in losses.items()},
              "groups": {}}
    for name, indices in groups.items():
        rows = {split: torch.stack([torch.cat([record[i] for i in indices]) for record in records]).double()
                for split, records in collected.items()}
        result["groups"][name] = {"train": agreement(rows["train"]), "heldout": agreement(rows["heldout"]),
                                   "train_heldout_mean_gradient_cosine": cosine(rows["train"].mean(0), rows["heldout"].mean(0))}
    return result
