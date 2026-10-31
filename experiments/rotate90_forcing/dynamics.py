"""Independent scalar response for the actual Schedule-Free two-buffer update.

Source: review/flat_spectrum.md, Gradient forcing, F1--F3. The homogeneous
matrix is shared across coordinates. Normalized gradients use the actual
changing raw second moment; the response is not a forecast of those gradients.
"""

from pathlib import Path
import runpy

import torch
from torch.nn import functional as F

from lab.infrastructure.benchmarks import build_task
from lab.infrastructure.benchmarks.samplers import gather

MEMORY = runpy.run_path(str(Path(__file__).resolve().parents[1] / "rotate90_transfer/memory.py"))
restore, coordinates = MEMORY["restore"], MEMORY["coordinates"]


def clock_step(clock):
    """F1's scalar matrix and normalized-gradient gains; advance only this copy."""
    beta = clock["betas"][0]
    clock["step"] += 1
    eta = clock["lr"] * (min(1., clock["step"] / clock["warmup"]) if clock["warmup"] else 1.)
    clock["maximum"] = max(eta, clock["maximum"])
    clock["total"] += clock["maximum"] ** 2
    c = clock["maximum"] ** 2 / clock["total"] if clock["total"] else 0.
    h = beta * (1 - c) - 1
    decay = clock["weight_decay"]
    return ((1 - c + eta * decay * h, c), (-eta * decay, 1.)), (eta * h, -eta)


def matrix_product(a, b):
    return tuple(tuple(sum(a[i][k] * b[k][j] for k in range(2)) for j in range(2)) for i in range(2))


def homogeneous(group, updates):
    clock = dict(group)
    result = ((1., 0.), (0., 1.))
    for _ in range(updates):
        matrix, _ = clock_step(clock)
        result = matrix_product(matrix, result)
    return result


def apply_matrix(matrix, y, z):
    return ([matrix[0][0] * a + matrix[0][1] * b for a, b in zip(y, z, strict=True)],
            [matrix[1][0] * a + matrix[1][1] * b for a, b in zip(y, z, strict=True)])


def evaluation(y, z, beta):
    return [a + (1 - 1 / beta) * (b - a) for a, b in zip(y, z, strict=True)]


def flatten(values):
    return torch.cat([v.detach().cpu().double().reshape(-1) for v in values])


def plane_residual(values, y, z):
    """Relative Euclidean distance to the initial query/base linear span."""
    value = flatten(values)
    basis = torch.stack((flatten(y), flatten(z)), 1)
    solution = torch.linalg.lstsq(basis, value, driver="gelsd")
    residual = value - basis @ solution.solution
    return {"relative_residual": float(residual.norm() / value.norm().clamp_min(1e-300)),
            "basis_rank": int(solution.rank)}


def max_error(left, right):
    return max(float((a - b).abs().max()) for a, b in zip(left, right, strict=True))


def check_source(spec, checkpoint, updates=32):
    """Compare independently propagated homogeneous/forced buffers to source steps."""
    model, optimizer, state = restore(spec, checkpoint)
    zero_model, zero_optimizer, _ = restore(spec, checkpoint)
    task = build_task(spec.benchmark, spec.seeds.data, torch.device("cpu"))
    sampler = task.sampler(spec.budget.batch, spec.seeds.batch_seed)
    sampler.restore(state)
    x, y, z = coordinates(optimizer)
    passive_y, passive_z = [v.clone() for v in y], [v.clone() for v in z]
    response_y, response_z = [torch.zeros_like(v) for v in y], [torch.zeros_like(v) for v in z]
    before_second = [zero_optimizer.state[p]["second"].clone() for p in zero_model.parameters()]
    active_error, passive_error = 0., 0.
    for _ in range(updates):
        batch = task.inputs(gather(sampler.next()[0]))
        optimizer.zero_grad(set_to_none=True)
        F.cross_entropy(model(batch[:, :2]), batch[:, 2]).backward()
        group = optimizer.param_groups[0]
        clock = dict(group)
        matrix, gain = clock_step(clock)
        with torch.no_grad():
            nu = []
            beta2 = group["betas"][1]
            for p in model.parameters():
                second = beta2 * optimizer.state[p]["second"] + (1 - beta2) * p.grad.square()
                nu.append(p.grad / ((second / (1 - beta2 ** clock["step"])).sqrt() + group["eps"]))
            passive_y, passive_z = apply_matrix(matrix, passive_y, passive_z)
            response_y, response_z = apply_matrix(matrix, response_y, response_z)
            response_y = [v + gain[0] * g for v, g in zip(response_y, nu, strict=True)]
            response_z = [v + gain[1] * g for v, g in zip(response_z, nu, strict=True)]
        optimizer.step()
        for p in zero_model.parameters():
            p.grad = torch.zeros_like(p)
        zero_optimizer.step()
        _, actual_y, actual_z = coordinates(optimizer)
        _, zero_y, zero_z = coordinates(zero_optimizer)
        active_error = max(active_error,
            max_error(actual_y, [a + b for a, b in zip(passive_y, response_y, strict=True)]),
            max_error(actual_z, [a + b for a, b in zip(passive_z, response_z, strict=True)]))
        passive_error = max(passive_error, max_error(zero_y, passive_y), max_error(zero_z, passive_z))
    beta = optimizer.param_groups[0]["betas"][0]
    actual_x, _, _ = coordinates(optimizer)
    predicted_x = [a + b for a, b in zip(evaluation(passive_y, passive_z, beta),
                                        evaluation(response_y, response_z, beta), strict=True)]
    change = sum(float((zero_optimizer.state[p]["second"] - v).square().sum())
                 for p, v in zip(zero_model.parameters(), before_second, strict=True)) ** .5
    return {"updates": updates, "forced_buffer_max_error": active_error,
            "homogeneous_buffer_max_error": passive_error,
            "evaluation_max_error": max_error(actual_x, predicted_x),
            "zero_gradient_second_change": change,
            "zero_gradient_plane": plane_residual(evaluation(zero_y, zero_z, beta), y, z),
            "response_evaluation_norm": float(flatten(evaluation(response_y, response_z, beta)).norm())}
