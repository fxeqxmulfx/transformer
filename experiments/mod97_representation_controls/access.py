"""Measure division-target access in the actual scalar MLP tangent maps.

Source: landscape_grokking_plan.md, M9.4, "Mechanisms to test". This is a
CPU float64 diagnostic of the exact stored float32 coordinates, at the
original random start and final checkpoint. It changes no training run.
All equations, including zero numerators, enter the measurement. Double
centering removes additive operand effects from both target and Jacobian;
this is a diagnostic projection, not a changed network or objective.

Run from python/ with uv run --locked python
../experiments/mod97_representation_controls/access.py.
"""

import hashlib
import json
from pathlib import Path

import torch
from torch.func import functional_call, jacrev

from lab.domain.paired import VectorReadout
from lab.domain.spec import describe, fingerprint
from lab.infrastructure.benchmarks import build_task
from lab.infrastructure.loader import load
from lab.infrastructure.nn import build_model
from lab.infrastructure.provenance import code
from lab.infrastructure.store import RunDirectories

HERE = Path(__file__).resolve().parent


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def interaction(table):
    """Orthogonal complement of the additive row/column prediction space."""
    return table - table.mean(0, keepdim=True) - table.mean(1, keepdim=True) + table.mean((0, 1), keepdim=True)


def prediction_jacobian(model, pairs, chunk=64):
    """Every scalar prediction derivative, in named_parameters order, on CPU."""
    parameters = dict(model.named_parameters())

    width = sum(value.numel() for value in parameters.values())
    jacobian = torch.empty((len(pairs), width), dtype=next(model.parameters()).dtype)
    for start in range(0, len(pairs), chunk):
        def predictions(weights):
            return functional_call(model, weights, (pairs[start:start + chunk],))

        gradients = jacrev(predictions)(parameters)
        jacobian[start:start + chunk] = torch.cat(
            [value.detach().flatten(1) for value in gradients.values()], dim=1)
    return jacobian


def linearized_path(kernel, cross_kernel, residual, steps):
    """Frozen full-batch GD with step 1/lambda_max; retain modes above 1e-12.

    Kernels include the training-example divisor. This is a finite-matrix
    counterfactual, not a model of the experiment's minibatch AdamW steps.
    """
    eigenvalues, vectors = torch.linalg.eigh(kernel)
    largest = float(eigenvalues[-1])
    assert largest > 0 and float(eigenvalues[0]) >= -1e-10 * largest
    retained = eigenvalues > 1e-12 * largest
    ratios = torch.where(retained, eigenvalues / largest, 0.0)
    coordinates = vectors.T @ residual
    path = []
    for step in steps:
        logarithm = step * torch.log1p(-ratios) if step else torch.zeros_like(ratios)
        decay = logarithm.exp()
        gain = torch.where(retained, -logarithm.expm1() / eigenvalues, 0.0)
        path.append((vectors @ (decay * coordinates),
                     -cross_kernel @ (vectors @ (gain * coordinates))))
    return eigenvalues, retained, path


def frozen_subset(jacobian, predictions, targets, indices):
    """Keep the first 512 source training and held-out rows, fixed before fitting."""
    train, heldout = (indices[name] for name in ("train", "heldout"))
    selected, count = jacobian[train], len(train)
    kernel = selected @ selected.T / count
    cross = jacobian[heldout] @ selected.T / count
    predictions, targets = predictions.flatten(), targets.flatten()
    steps = (0,) + tuple(10 ** power for power in range(13))
    values, retained, path = linearized_path(kernel, cross, predictions[train] - targets[train], steps)

    def metrics(output, target):
        numeric_error = (output - target) * 48
        correct = ((output + 1) * 48).round() == ((target + 1) * 48).round()
        return {"half_mse": float((output - target).square().mean() / 2),
                "continuous_numeric_mse": float(numeric_error.square().mean()),
                "accuracy": float(correct.double().mean())}

    return {"training_examples": count, "heldout_examples": len(heldout),
            "step": 1 / float(values[-1]), "relative_eigenvalue_cutoff": 1e-12,
            "retained_rank": int(retained.sum()), "kernel_eigenvalues": values.tolist(),
            "trajectory": [{"updates": step, "train": metrics(targets[train] + residual, targets[train]),
                            "heldout": metrics(predictions[heldout] + change, targets[heldout])}
                           for step, (residual, change) in zip(steps, path, strict=True)]}


def summarize(jacobian, predictions, targets, model):
    """Exact finite-matrix contractions; no spectral truncation or inversion.

    For K = (P J)(P J)^T / N and interaction target y, the target Rayleigh
    quotient is ||J^T y||^2 / (N ||y||^2). Its ratio to trace(K) removes
    a common Jacobian scale. The same quantities are measured for P(f-y).
    These are local Euclidean tangent quantities, not an AdamW step model.
    """
    rows = targets.numel()
    raw_energy = float(torch.linalg.vector_norm(jacobian).square())
    cube = jacobian.reshape(*targets.shape, -1)
    row_mean, column_mean = cube.mean(1, keepdim=True), cube.mean(0, keepdim=True)
    grand_mean = row_mean.mean(0, keepdim=True)
    cube.sub_(row_mean).sub_(column_mean).add_(grand_mean)
    energy = float(torch.linalg.vector_norm(jacobian).square())
    components = {"target": interaction(targets).flatten(),
                  "residual": interaction(predictions - targets).flatten()}
    contractions = {name: jacobian.T @ vector for name, vector in components.items()}
    result = {"examples": rows, "parameters": jacobian.shape[1],
              "raw_jacobian_energy": raw_energy, "interaction_jacobian_energy": energy,
              "interaction_jacobian_energy_fraction": energy / raw_energy if raw_energy else None,
              "interaction_kernel_trace": energy / rows,
              "target_interaction_energy_fraction": float(components["target"].square().sum() /
                                                          targets.square().sum()),
              "full_half_mse": float((predictions - targets).square().mean() / 2),
              "layers": {}}
    for name, vector in components.items():
        squared = float(vector.square().sum())
        contraction = float(contractions[name].square().sum())
        result[name] = {"interaction_energy": squared,
                        "jacobian_target_contraction_squared": contraction,
                        "kernel_rayleigh": contraction / (rows * squared) if squared else None,
                        "trace_normalized_alignment": contraction / (energy * squared)
                        if energy and squared else None}
    offset = 0
    for name, parameter in model.named_parameters():
        end = offset + parameter.numel()
        result["layers"][name] = {
            "interaction_jacobian_energy": float(torch.linalg.vector_norm(jacobian[:, offset:end]).square()),
            **{component + "_contraction_squared": float(vector[offset:end].square().sum())
               for component, vector in contractions.items()}}
        offset = end
    assert offset == jacobian.shape[1]
    return result


def measure(model, pairs, targets, indices=None):
    with torch.no_grad():
        predictions = model(pairs).reshape_as(targets)
    jacobian = prediction_jacobian(model, pairs)
    frozen = None if indices is None else frozen_subset(jacobian, predictions, targets, indices)
    result = summarize(jacobian, predictions, targets, model)
    if frozen is not None:
        result["frozen_subset"] = frozen
    assert all(value is None or 0 <= value <= 1 + 1e-12
               for value in (result[name]["trace_normalized_alignment"] for name in ("target", "residual")))
    return result


def main():
    torch.set_num_threads(4)
    study, runs = load(HERE), RunDirectories(HERE / "runs")
    selected = {label: spec for label, spec in study.experiments.items()
                if not isinstance(spec.model, VectorReadout)}
    assert len(selected) == 12
    # Fail before allocating Jacobians if the paired cohort is incomplete.
    assert all(runs.open(label).result() is not None and
               runs.open(label).result()["updates"] == spec.budget.updates
               for label, spec in selected.items()), "Complete all 12 scalar-output runs first"
    current_code, measurements = code(), {}
    for label, spec in selected.items():
        run = runs.open(label)
        assert run.manifest()["description"] == describe(spec)
        provenance = run.provenance()
        assert provenance["engine"]["code"] == current_code
        assert provenance["engine"]["torch"] == torch.__version__
        assert provenance["commit"]["lab_modified"] is False
        task = build_task(spec.benchmark, spec.seeds.data, torch.device("cpu"), spec.model)
        assert task.vocab == 97, "Numeric diagnostic units are fixed to the original modulus"
        rows = torch.cat(tuple(task.splits.values()))
        rows = rows[(rows[:, 0] * task.vocab + rows[:, 1]).argsort()]
        assert len(rows) == task.vocab * (task.vocab - 1)
        assert bool(((rows[:, 1] * rows[:, 2] - rows[:, 0]) % task.vocab == 0).all())
        targets = (rows[:, 2].double() * (2 / (task.vocab - 1)) - 1).reshape(task.vocab, task.vocab - 1)
        indices = {name: part[:512, 0] * (task.vocab - 1) + part[:512, 1] - 1
                   for name, part in task.splits.items()}
        model = build_model(spec.model, task.vocab, spec.seeds.model).double()
        initial = measure(model, rows[:, :2], targets, indices)
        checkpoint = run.checkpoint("cpu")
        assert checkpoint["step"] == spec.budget.updates
        model.load_state_dict(checkpoint["model"])
        final = measure(model, rows[:, :2], targets, indices)
        measurements[label] = {"experiment_fingerprint": fingerprint(spec), "split": task.summary(),
                               "checkpoint_sha256": sha256(run.path / "checkpoint.pt"),
                               "training_engine": provenance["engine"], "initial": initial, "final": final}
        print(label, "target alignment", initial["target"]["trace_normalized_alignment"],
              final["target"]["trace_normalized_alignment"], flush=True)
    output = {"analysis_script_sha256": sha256(Path(__file__)), "analysis_device": "cpu",
              "analysis_dtype": "float64", "torch": torch.__version__,
              "scope": "full-table tangent diagnostic; no training intervention or AdamW equivalence",
              "runs": measurements}
    (HERE / "access_results.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
