"""Construct a bounded SwiGLU division witness in the actual tied model.

M9.4b of landscape_grokking_plan.md. This is a representation certificate,
not a trained checkpoint or a claim about random-start attraction. A pair
of opposite SiLU gates implements an exact product in real arithmetic:
    (SiLU(t) - SiLU(-t)) * u = t * u.
Nine such products cancel the unwanted skip coordinates and form a faithful
character of a/b. Both RMSNorms, shared E, zero residue and residual remain.
Run from python/: uv run --locked python ../experiments/rotate90_analytic/expressivity.py.
"""

import hashlib
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from lab.dsl import LabDivision, Rotate90, SwiGLU
from lab.infrastructure.benchmarks import build_task
from lab.infrastructure.nn import build_model
from lab.infrastructure.provenance import provenance

HERE = Path(__file__).resolve().parent
PRIME, WIDTH, HIDDEN, EPSILON, CONSTANT = 97, 128, 43, 1e-5, 0.1


def primitive_root(prime):
    return next(g for g in range(2, prime)
                if len({pow(g, k, prime) for k in range(prime - 1)}) == prime - 1)


@torch.no_grad()
def construct(dtype, device):
    model = build_model(Rotate90(WIDTH, SwiGLU(HIDDEN), EPSILON), PRIME, 11).to(device=device, dtype=dtype)
    for parameter in model.parameters():
        parameter.zero_()
    generator = primitive_root(PRIME)
    # Every E row has squared norm 1+c^2. Even coordinates form U and J U
    # occupies disjoint odd coordinates, so x's squared norm is 2(1+c^2).
    model.E[:, 0] = CONSTANT
    model.E[0, 6] = 1
    for k in range(PRIME - 1):
        residue = pow(generator, k, PRIME)
        angle = 2 * math.pi * k / (PRIME - 1)
        model.E[residue, 2] = math.cos(angle)
        model.E[residue, 4] = math.sin(angle)
    inverse_alpha_squared = EPSILON + 2 * (1 + CONSTANT ** 2) / WIDTH
    # r0 is alpha*c. The first five products subtract b's constant/circle
    # and a's circle from the skip; a's constant and zero flag remain.
    terms = [(0, 0, 1, -inverse_alpha_squared / CONSTANT),
             (0, 2, 2, -inverse_alpha_squared / CONSTANT),
             (0, 4, 4, -inverse_alpha_squared / CONSTANT),
             (0, 3, 3, -inverse_alpha_squared / CONSTANT),
             (0, 5, 5, -inverse_alpha_squared / CONSTANT),
             (2, 3, 2, inverse_alpha_squared),
             (4, 5, 2, inverse_alpha_squared),
             (4, 3, 4, inverse_alpha_squared),
             (2, 5, 4, -inverse_alpha_squared)]
    for term, (gate, value, output, coefficient) in enumerate(terms):
        for offset, sign in enumerate((1, -1)):
            column = 2 * term + offset
            model.G[gate, column] = sign
            model.U[value, column] = 1
            model.D[column, output] = sign * coefficient
    assert 2 * len(terms) == 18 <= HIDDEN
    assert all(bool(torch.isfinite(p).all()) for p in model.parameters())
    assert max(float(p.abs().max()) for p in model.parameters()) <= 1
    return model, generator


@torch.no_grad()
def check(dtype, device, rows):
    model, generator = construct(dtype, device)
    rows = rows.to(device)
    a, b, answer = rows.T
    assert bool(((b * answer - a) % PRIME == 0).all())
    rotated = torch.stack((-model.E[b, 1::2], model.E[b, 0::2]), dim=-1).flatten(1)
    x = model.E[a] + rotated
    r = model.normalize(x)
    state = x + (F.silu(r @ model.G) * (r @ model.U)) @ model.D
    expected_state = model.E[answer]
    logits = model(rows[:, :2])
    expected_logits = model.normalize(expected_state) @ model.E.T
    state_error = float((state - expected_state).abs().max())
    logit_error = float((logits - expected_logits).abs().max())
    tolerance = 1e-11 if dtype == torch.float64 else 2e-5
    assert state_error < tolerance and logit_error < tolerance
    assert bool((logits.argmax(dim=-1) == answer).all())
    correct = logits.gather(1, answer[:, None]).squeeze(1)
    competitors = logits.clone().scatter_(1, answer[:, None], -torch.inf).max(dim=-1).values
    margin = float((correct - competitors).min())
    assert margin > 0
    expected_margin = (1 - math.cos(2 * math.pi / (PRIME - 1))) / math.sqrt(
        EPSILON + (1 + CONSTANT ** 2) / WIDTH)
    assert abs(margin - expected_margin) < tolerance
    parameter_hashes = {name: hashlib.sha256(p.detach().cpu().numpy().tobytes()).hexdigest()
                        for name, p in model.named_parameters()}
    return {"dtype": str(dtype), "device": str(device), "primitive_root": generator,
            "examples": len(rows), "zero_numerator_examples": int((a == 0).sum()),
            "accuracy": float((logits.argmax(dim=-1) == answer).double().mean()),
            "minimum_margin": margin, "expected_minimum_margin": expected_margin,
            "cross_entropy": float(F.cross_entropy(logits, answer)),
            "max_state_error": state_error, "max_logit_error": logit_error,
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "max_parameter_abs": max(float(p.abs().max()) for p in model.parameters()),
            "active_gates": 18, "available_gates": HIDDEN,
            "parameter_sha256": parameter_hashes, "runtime": provenance(device)}


def main():
    torch.set_num_threads(1)
    task = build_task(LabDivision(), 11, torch.device("cpu"))
    rows = torch.cat(tuple(task.splits.values()))
    assert len(rows) == PRIME * (PRIME - 1)
    assert len(torch.unique(rows[:, 0] * PRIME + rows[:, 1])) == len(rows)
    checks = [check(dtype, torch.device("cpu"), rows) for dtype in (torch.float64, torch.float32)]
    if torch.cuda.is_available():
        checks.append(check(torch.float32, torch.device("cuda"), rows))
    payload = {"source_revision": "f49969393a7482490d45483d8a4fd3046b1adede",
               "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "width": WIDTH, "hidden": HIDDEN, "epsilon": EPSILON, "constant_feature": CONSTANT,
               "claim": "Finite constructive representation, not a training or zero-loss certificate",
               "checks": checks}
    (HERE / "division_witness.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    for result in checks:
        print({key: value for key, value in result.items() if key not in ("runtime", "parameter_sha256")}, flush=True)


if __name__ == "__main__":
    main()
