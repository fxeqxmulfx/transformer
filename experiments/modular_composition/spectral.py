"""Numerical checks of the precise local spectral statements, no training."""

import json
from pathlib import Path

import torch


def check(prime):
    a, b = torch.arange(prime, dtype=torch.int64)[:, None], torch.arange(prime, dtype=torch.int64)[None, :]
    labels = (a**3 + a * b) % prime
    kernel = torch.exp(2j * torch.pi * labels.to(torch.float64) / prime)
    gram_error = float((kernel @ kernel.mH - prime * torch.eye(prime)).abs().max())
    singular = torch.linalg.svdvals(kernel)
    singular_error = float((singular - prime**.5).abs().max())
    shifted = torch.exp(2j * torch.pi * ((labels + 1) % prime).to(torch.float64) / prime)
    phase = torch.exp(torch.tensor(2j * torch.pi / prime, dtype=torch.complex128))
    shift_error = float((shifted - kernel * phase).abs().max())
    alternative = kernel.clone()
    alternative[0, 0] *= -1
    true_norm, wrong_norm = float(singular.sum()), float(torch.linalg.svdvals(alternative).sum())
    generator = next(g for g in range(1, prime) if len({pow(g, i, prime) for i in range(prime - 1)}) == prime - 1)
    logs = {pow(generator, i, prime): i for i in range(prime - 1)}
    phases = torch.tensor([[logs[a * pow(b, -1, prime) % prime] for b in range(1, prime)]
                          for a in range(1, prime)], dtype=torch.float64)
    division = torch.exp(2j * torch.pi * phases / (prime - 1))
    division_singular = torch.linalg.svdvals(division)
    rank_one_error = float(division_singular[1:].max())
    assert gram_error < 1e-9 and singular_error < 1e-9 and shift_error < 1e-9
    assert wrong_norm < true_norm and rank_one_error < 1e-9
    return {"prime": prime, "mixed_gram_max_error": gram_error,
            "mixed_singular_max_error": singular_error, "constant_shift_max_error": shift_error,
            "true_mixed_nuclear_norm": true_norm, "one_wrong_entry_nuclear_norm": wrong_norm,
            "division_second_singular_value": rank_one_error}


def main():
    torch.set_num_threads(1)
    results = {"scope": "Local two-variable kernels only; numerical checks accompany existing Lean theorems.",
               "cases": [check(prime) for prime in (17, 97, 193)]}
    (Path(__file__).parent / "spectral_checks.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results))


if __name__ == "__main__":
    main()
