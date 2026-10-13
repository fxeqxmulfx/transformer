"""Head and train-selected subspace ablations on saved model copies.

Source: Nanda et al., arXiv:2301.05217v1, sections 4 and 5.1, causal
removal of components. Deviations: remove every individual head before its
output projection, after XSA; compare top/bottom eight centered train
activation PCs and a matched random rank-eight subspace at the answer
query. This establishes sensitivity to these interventions, not a unique
circuit or a guarantee that an off-manifold intervention is natural.
"""

from contextlib import contextmanager

import torch
from torch.nn import functional as F

from .capture import evaluating, forward
from .features import spectrum


def metrics(logits, targets, train, heldout):
    result = {}
    for name, selected in (("train", train), ("heldout", heldout)):
        x, y = logits[selected].double(), targets[selected]
        result[name] = {"accuracy": float((x.argmax(-1) == y).double().mean()),
                        "loss": float(F.cross_entropy(x, y))}
    return result


@contextmanager
def remove_head(attention, head, heads):
    """Zero the actual merged head output at all positions; retain biases."""
    def hook(module, inputs):
        x = inputs[0].clone()
        width = x.shape[-1] // heads
        x[..., head * width:(head + 1) * width] = 0
        return (x, *inputs[1:])

    handle = attention.output.register_forward_pre_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def remove_subspace(block, mean, directions):
    """Remove centered directions at equals; preserve train feature mean."""
    def hook(module, inputs, output):
        x = output.clone()
        centered = x[:, 4].double() - mean.to(x.device)
        basis = directions.to(x.device)
        correction = centered @ basis @ basis.T
        x[:, 4] -= correction.to(x.dtype)
        return x

    handle = block.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def measure(model, inputs, targets, train, heldout, features, logits, heads, batch=512):
    baseline = metrics(logits, targets, train, heldout)
    result = {"baseline": baseline, "heads": {}, "subspaces": {}, "rank": 8,
              "subspace_fit_split": "train_only", "head_positions": "all_prompt_positions",
              "subspace_position": "answer_query_equals"}

    def effect(changed):
        observed = metrics(changed, targets, train, heldout)
        for name, selected in (("train", train), ("heldout", heldout)):
            observed[name]["loss_increase"] = observed[name]["loss"] - baseline[name]["loss"]
            observed[name]["accuracy_drop"] = baseline[name]["accuracy"] - observed[name]["accuracy"]
            observed[name]["prediction_change_fraction"] = float((changed[selected].argmax(-1) != logits[selected].argmax(-1)).double().mean())
        return observed

    with evaluating(model):
        for index, block in enumerate(model.blocks):
            for head in range(heads):
                with remove_head(block.attention, head, heads):
                    changed = forward(model, inputs, batch)
                result["heads"][f"blocks.{index}.head{head}"] = effect(changed)
            stats, mean, directions = spectrum(features[f"blocks.{index}"][train])
            rank = min(8, directions.shape[0])
            generator = torch.Generator().manual_seed(2718 + index)
            random = torch.linalg.qr(torch.randn(directions.shape[0], rank, generator=generator, dtype=torch.float64)).Q
            choices = {"top": directions[:, :rank], "bottom": directions[:, -rank:], "random": random}
            for name, basis in choices.items():
                with remove_subspace(block, mean, basis):
                    changed = forward(model, inputs, batch)
                result["subspaces"][f"blocks.{index}.{name}"] = {
                    "removed_train_variance_fraction": (sum(stats["eigenvalue_energy_fractions"][:rank]) if name == "top" else
                        sum(stats["eigenvalue_energy_fractions"][-rank:]) if name == "bottom" else None),
                    **effect(changed)}
    return result
