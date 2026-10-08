"""Actual CE curves along the disposable native CPU displacement.

Source: native endpoint and sampler protocol in momentum_directions at
444b4ad; PyTorch 2.14.1 autograd Hessian-vector products. Lean comparison:
Grokking.AdamW.CurvatureBound and CurvatureCounterexample. Deviations:
exhaustive training mean CE and held-out nonzero answer CE are read on
fixed interpolations between observed CPU endpoints. Rates are diagnostic
only. Observed float32 endpoints are interpolated in float64 then rounded
back to the model dtype; HVP directions are cast to that dtype. Sampled
curvature neither bounds the whole interval nor certifies a safe rate.
"""

import copy
import hashlib
import math

import torch

from lab.domain.optimizers import clipping
from lab.domain.training import rate
from lab.infrastructure.benchmarks.samplers import gather
from lab.infrastructure.optim import build_optimizer

from momentum_directions import flatten, retained_direction, scopes
from probes.capture import evaluating

FRACTIONS = (0., .01, .1, .25, .5, .75, 1.)
HESSIAN_FRACTIONS = (0., .25, .5, .75, 1.)


def endpoint(model, task, spec, snapshot):
    """One isolated native step from the actual retained moment/sampler state."""
    clone = copy.deepcopy(model)
    parameters = [p for p in clone.parameters() if p.requires_grad]
    optimizer = build_optimizer(spec.optimizer, clone)
    if [id(p) for g in optimizer.param_groups for p in g["params"]] != [id(p) for p in parameters]:
        raise ValueError("Native groups must follow unique model registration order")
    ids = [g["params"] for g in optimizer.state_dict()["param_groups"]]
    if [g["params"] for g in snapshot["optimizer"]["param_groups"]] != ids:
        raise ValueError("Archived native parameter registration differs")
    optimizer.load_state_dict(copy.deepcopy(snapshot["optimizer"]))
    retained_direction(optimizer, snapshot["step"])
    before = [p.detach().double().clone() for p in parameters]
    sampler = task.sampler(spec.budget.batch, spec.seeds.batch_seed)
    sampler.restore(copy.deepcopy(snapshot))
    parts, place = sampler.next()
    indices = gather(parts)
    learning_rate = rate(spec.optimizer.lr, spec.schedule, snapshot["step"])
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Positive scheduled probe rate is required")
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
    clone.train()
    optimizer.zero_grad(set_to_none=True)
    output, targets = task.forward(clone, task.inputs(indices))
    task.loss(output, targets).backward()
    norm = torch.nn.utils.clip_grad_norm_(clone.parameters(), clipping(spec.optimizer))
    if not torch.isfinite(norm):
        raise FloatingPointError("Nonfinite restored-minibatch gradient")
    optimizer.step()
    retained_direction(optimizer, snapshot["step"] + 1)
    after = [p.detach().double().clone() for p in parameters]
    if not all(torch.isfinite(value).all() for value in before + after):
        raise FloatingPointError("Nonfinite native parameter endpoint")
    return clone, before, after, learning_rate, {"examples": len(indices), "place": place,
        "indices_sha256": hashlib.sha256(indices.numpy().astype("<i8").tobytes()).hexdigest()}


def objective(task, output, targets, kind):
    if kind == "full":
        return task.loss(output, targets)
    if kind == "answer":
        return task.position_losses(output, targets)[0]
    raise ValueError("Choose the original full CE or answer-only CE")


def point(model, task, rows, direction, kind, batch=512, hessian=True):
    """Sample-weighted loss, slope and HVP; no parameter .grad is changed."""
    if len(rows) == 0 or batch < 1:
        raise ValueError("Nonempty rows and positive chunk size are required")
    parameters = [p for p in model.parameters() if p.requires_grad]
    if len(direction) != len(parameters) or any(d.shape != p.shape for p, d in zip(parameters, direction)):
        raise ValueError("Direction must match unique parameter registration and shapes")
    typed = [d.to(p.dtype) for p, d in zip(parameters, direction)]
    loss_sum, correct, slope, curvature = 0., 0, 0., 0.
    mean_gradient = torch.zeros(sum(p.numel() for p in parameters), dtype=torch.float64)
    with evaluating(model), torch.enable_grad():
        for chunk in rows.split(batch):
            output, targets = task.forward(model, chunk)
            loss = objective(task, output, targets, kind)
            weight = len(chunk) / len(rows)
            loss_sum += float(loss.detach()) * weight
            correct += int((output[:, 0].argmax(-1) == targets[:, 0]).sum())
            first = torch.autograd.grad(loss, parameters, create_graph=hessian, allow_unused=True)
            dense = [torch.zeros_like(p) if g is None else g for p, g in zip(parameters, first)]
            mean_gradient.add_(flatten(dense), alpha=weight)
            slope -= sum(float((g.detach().double() * d).sum()) for g, d in zip(dense, direction)) * weight
            if hessian:
                terms = [(g * d).sum() for g, d in zip(first, typed) if g is not None and g.requires_grad]
                if terms:
                    linear = sum(terms)
                    hvp = torch.autograd.grad(linear, parameters, allow_unused=True)
                    curvature += sum(float((h.detach().double() * d.double()).sum())
                                     for h, d in zip(hvp, typed) if h is not None) * weight
    values = (loss_sum, slope, curvature, float(mean_gradient.norm()))
    if not all(math.isfinite(value) for value in values):
        raise FloatingPointError("Nonfinite loss/gradient/curvature observation")
    return {"examples": len(rows), "loss": loss_sum, "answer_accuracy": correct / len(rows),
            "rate_derivative": slope, "rate_curvature": curvature if hessian else None,
            "gradient_l2": values[-1]}


@torch.no_grad()
def interpolate(model, before, after, fraction):
    """Round the same endpoint interpolation; fractions zero and one are exact."""
    parameters = [p for p in model.parameters() if p.requires_grad]
    for parameter, old, new in zip(parameters, before, after):
        value = old if fraction == 0 else new if fraction == 1 else old + fraction * (new - old)
        parameter.copy_(value.to(parameter.dtype))


def measure(model, task, spec, snapshot, batch=512):
    """Current CE curves only; preserve supplied weights, state, modes and RNG."""
    if snapshot["step"] < 1 or any(p.device.type != "cpu" for p in model.parameters()):
        raise ValueError("Actual noninitial native states and a CPU model are required")
    with evaluating(model), torch.random.fork_rng(devices=[]):
        clone, before, after, learning_rate, minibatch = endpoint(model, task, spec, snapshot)
        direction = [(old - new) / learning_rate for old, new in zip(before, after)]
        parameters = [p for p in clone.parameters() if p.requires_grad]
        cast = [d.to(p.dtype).double() for p, d in zip(parameters, direction)]
        flat = flatten(direction)
        populations = scopes(task)
        selections = {"train_all": "full", "heldout_nonzero": "answer"}
        result = {"protocol": "fixed_observed_native_CPU_displacement; exhaustive_CE_curves",
                  "precision": "float64_endpoint_interpolation_rounded_to_model_dtype; native_dtype_HVP; uncertified",
                  "rate": learning_rate, "minibatch": minibatch,
                  "direction_l2": float(flat.norm()),
                  "HVP_direction_cast_relative_error": float((flatten(cast) - flat).norm() / flat.norm())
                    if float(flat.norm()) > 0 else None,
                  "fractions": list(FRACTIONS), "hessian_fractions": list(HESSIAN_FRACTIONS), "populations": {}}
        for name, kind in selections.items():
            records = []
            for fraction in FRACTIONS:
                interpolate(clone, before, after, fraction)
                observed = point(clone, task, populations[name], direction, kind, batch, fraction in HESSIAN_FRACTIONS)
                records.append({"fraction": fraction, "rate": learning_rate * fraction, **observed})
            initial = records[0]
            for record in records:
                eta = record["rate"]
                record.update(loss_change=record["loss"] - initial["loss"],
                              linear_prediction=eta * initial["rate_derivative"],
                              quadratic_prediction=eta * initial["rate_derivative"] + eta ** 2 * initial["rate_curvature"] / 2)
                record["quadratic_remainder"] = record["loss_change"] - record["quadratic_prediction"]
            result["populations"][name] = {"objective": kind, "records": records,
                "initial_quadratic_crossing_rate": -2 * initial["rate_derivative"] / initial["rate_curvature"]
                   if initial["rate_derivative"] < 0 and initial["rate_curvature"] > 0 else None,
                "interpretation": "sampled_HVP_not_an_interval_bound; crossing_rate_not_certified"}
    return result
