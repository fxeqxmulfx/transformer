"""Separate answer and EOS gradients without changing the training objective.

Source: Power et al.'s openai/grok at commit
3d64b1d8c1d595dd8ebdb7771998823f1b14c7b3, via ModularTask at 43d4d66.
Deviation: read three gradients on the same forward pass of an offline
model: answer CE, EOS CE, and the actual mean of both supervised positions.
The EOS position sees the supplied answer, as in training. A shared target
is a possible explanation of alignment, not an assumed cause of grokking.

The two component gradients need not be orthogonal. Their cross term is
signed; reported contributions are not nonnegative energy fractions.
"""

import hashlib

import torch

from probes.capture import evaluating
from probes.gradients import agreement, cosine, group


def energy_terms(answer, eos, full):
    """Mean batch energies with a signed cross term and measured residuals."""
    answer, eos, full = (value.double() for value in (answer, eos, full))
    reconstruction = (answer + eos) / 2
    residual = (full - reconstruction).norm(dim=1)
    scale = (answer.norm(dim=1) + eos.norm(dim=1)) / 2
    nonzero = scale > 0
    answer_energy = float(answer.square().sum(1).mean()) / 4
    eos_energy = float(eos.square().sum(1).mean()) / 4
    cross = float((answer * eos).sum(1).mean()) / 2
    actual = float(full.square().sum(1).mean())
    return {"full_squared_norm": actual, "quarter_answer_squared_norm": answer_energy,
            "quarter_EOS_squared_norm": eos_energy, "half_answer_EOS_dot": cross,
            "energy_identity_absolute_error": abs(actual - answer_energy - eos_energy - cross),
            "max_gradient_reconstruction_l2": float(residual.max()),
            "max_gradient_reconstruction_relative_error":
                float((residual[nonzero] / scale[nonzero]).max()) if nonzero.any() else None,
            "mean_answer_EOS_gradient_cosine": cosine(answer.mean(0), eos.mean(0))}


def split_dot_terms(train, heldout):
    """Decompose the split-mean dot product using its actual full denominator.

    Terms divided by that denominator sum to the full cosine up to floating
    point error. An individual term is not a cosine and can exceed one.
    """
    train, heldout = ({key: value.double().mean(0) for key, value in split.items()}
                      for split in (train, heldout))
    actual = float(train["full"] @ heldout["full"])
    denominator = float(train["full"].norm() * heldout["full"].norm())
    terms = {f"{left}_{right}": float(train[left] @ heldout[right]) / 4
             for left in ("answer", "EOS") for right in ("answer", "EOS")}
    return {"full_dot": actual, "quarter_component_dots": terms,
            "dot_identity_absolute_error": abs(actual - sum(terms.values())),
            "terms_over_full_norm_product":
                {key: value / denominator for key, value in terms.items()} if denominator > 0 else None}


def measure(model, task, batches=8, batch=32):
    """Autograd diagnostics on fixed disjoint batches, with tied weights once."""
    parameters = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    device = next(model.parameters()).device
    collected, losses, selections = {}, {}, {}
    with evaluating(model), torch.enable_grad():
        for split in ("train", "heldout"):
            data = task.splits[split]
            count = min(batches * batch, len(data))
            if count == 0:
                raise ValueError("Gradient comparison requires nonempty fixed batches")
            selected = torch.randperm(len(data), generator=torch.Generator().manual_seed(1729))[:count]
            selections[split] = {"examples": count,
                                 "indices_sha256": hashlib.sha256(selected.numpy().astype("<i8").tobytes()).hexdigest()}
            collected[split] = {key: [] for key in ("answer", "EOS", "full")}
            losses[split] = {key: [] for key in collected[split]}
            for indices in selected.split(batch):
                output, targets = task.forward(model, data[indices].to(device))
                if targets.ndim != 2 or targets.shape[1] != 2 or output.shape[:2] != targets.shape:
                    raise ValueError("Expected the original two supervised positions: answer and EOS")
                position = task.position_losses(output, targets)
                objectives = {"answer": position[0], "EOS": position[1], "full": task.loss(output, targets)}
                for key in ("full", "answer", "EOS"):
                    gradients = torch.autograd.grad(objectives[key], [p for name, p in parameters],
                                                    allow_unused=True, retain_graph=key != "EOS")
                    collected[split][key].append([
                        torch.zeros_like(p).cpu().flatten() if g is None else g.detach().cpu().flatten()
                        for (name, p), g in zip(parameters, gradients)])
                    losses[split][key].append(float(objectives[key].detach()))
    groups = {name: [i for i, (key, p) in enumerate(parameters) if group(key) == name]
              for name in sorted({group(key) for key, p in parameters})}
    groups["all"] = list(range(len(parameters)))
    result = {"objective": "actual_mean_answer_and_EOS_CE; separate_position_gradients_are_diagnostics",
              "supervision": "positions_4_and_5_of_the_six_token_training_input; EOS_sees_answer",
              "batch_size": batch, "batches_by_split": {key: len(rows["full"]) for key, rows in collected.items()},
              "sample_seed": 1729, "selection": selections, "unique_trainable_tensors": len(parameters),
              "loss_by_split": {split: {key: sum(values) / len(values) for key, values in rows.items()}
                                for split, rows in losses.items()}, "groups": {}}
    for name, indices in groups.items():
        rows = {split: {key: torch.stack([torch.cat([record[i] for i in indices])
                                         for record in records]).double() for key, records in values.items()}
                for split, values in collected.items()}
        result["groups"][name] = {
            "components": {key: {"train": agreement(rows["train"][key]),
                                 "heldout": agreement(rows["heldout"][key]),
                                 "train_heldout_mean_gradient_cosine":
                                     cosine(rows["train"][key].mean(0), rows["heldout"][key].mean(0))}
                           for key in ("answer", "EOS", "full")},
            "energy_by_split": {split: energy_terms(values["answer"], values["EOS"], values["full"])
                                for split, values in rows.items()},
            "train_heldout_dot_decomposition": split_dot_terms(rows["train"], rows["heldout"])}
    return result
