#!/usr/bin/env python3
"""Search for local minimizers of the phase energy V_full above n²/8.

`landscape_plan.md`, M3.4b.  `V_full(ψ) = Σ_{x, y} (1 - cos(ψ_x - ψ_y - ψ_{x-y}))`
over all pairs of `ℤ/n` (`Transformer.Landscape.PhaseModel.divisionEnergy` of the full
table).  Its zeros are the phases `ψ_x = 2πjx/n` of the characters up to
periods, and `Landscape.PhaseModel.Landscape` proves that for `n ≥ 4` every local
minimizer with `V_full < n²/8` is one.  This search asks whether a local
minimizer with `V_full ≥ n²/8` exists for `2 ≤ n ≤ 30`.

For every `n` it runs gradient descent, of step `1/(3n)` below the inverse of
the Hessian bound `‖Rᵀ diag(cos θ) R‖ ≤ ‖RᵀR‖ = 3n`, from five families of
starts: uniform phases, phases in `{0, π}`, phases in `(2π/m)ℤ` for
`m ∈ {3, 4, 5, 6, 8}`, two characters mixed on a random subset, and the
non-character critical points that Newton's method finds from uniform starts,
five times each.  Every point of `{0, π}ⁿ` is critical, and so are many
rational and mixed points, so every start but the uniform ones is moved by
`10⁻²` in a random direction.  Each final point is classified by its energy,
gradient and least Hessian eigenvalue.  The Newton critical points
are reported too: the lowest energy, and how many have no negative eigenvalue.

Run inside `python/`:  uv run --locked python ../scripts/phase_minima.py
"""

from __future__ import annotations

import argparse

import numpy as np

FAMILIES = ("uniform", "binary", "rational", "mixture", "saddle")


def rows(n: int) -> np.ndarray:
    """The rows `e_{a+b} - e_a - e_b` of the full table, one per pair `(a, b)`:
    the row `e_x - e_y - e_{x-y}` of the pair `(x, y) = (a + b, a)`."""
    a, b = (grid.ravel() for grid in np.meshgrid(np.arange(n), np.arange(n), indexing="ij"))
    table = np.zeros((n * n, n))
    np.add.at(table, (np.arange(n * n), (a + b) % n), 1.0)
    np.add.at(table, (np.arange(n * n), a), -1.0)
    np.add.at(table, (np.arange(n * n), b), -1.0)
    return table


def energy(psi: np.ndarray, table: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`V_full` and its gradient at each row of `psi`."""
    theta = psi @ table.T
    return (1 - np.cos(theta)).sum(axis=-1), np.sin(theta) @ table


def hessian(psi: np.ndarray, table: np.ndarray) -> np.ndarray:
    return (table * np.cos(table @ psi)[:, None]).T @ table


def newton_critical_points(n: int, table: np.ndarray, starts: int, rng) -> list[np.ndarray]:
    """Distinct non-character critical points reached by Newton's method."""
    found: dict[tuple[float, int], np.ndarray] = {}
    for _ in range(starts):
        psi = rng.uniform(0, 2 * np.pi, n)
        for _ in range(80):
            _, gradient = energy(psi, table)
            values, vectors = np.linalg.eigh(hessian(psi, table))
            values = np.where(np.abs(values) > 1e-9, values, 1e-9)
            psi = psi - vectors @ ((vectors.T @ gradient) / values)
        value, gradient = energy(psi, table)
        if np.linalg.norm(gradient) < 1e-9 and value > 1e-6:
            index = int((np.linalg.eigvalsh(hessian(psi, table)) < -1e-7).sum())
            found.setdefault((round(float(value), 6), index), np.mod(psi, 2 * np.pi))
    return [found[key] for key in sorted(found)]


def starts(n: int, count: int, saddles: list[np.ndarray], rng) -> tuple[np.ndarray, list[str]]:
    """The starting phases of every family, and the family of each."""
    x = np.arange(n)
    blocks = {
        "uniform": rng.uniform(0, 2 * np.pi, (count, n)),
        "binary": np.pi * rng.integers(0, 2, (count, n)),
        "rational": 2 * np.pi * rng.integers(0, 120, (count, n))
        / rng.choice([3, 4, 5, 6, 8], (count, 1)),
    }
    j, k = rng.integers(0, n, (2, count, 1))
    subset = rng.integers(0, 2, (count, n)).astype(bool)
    blocks["mixture"] = 2 * np.pi * np.where(subset, j * x, k * x) / n
    blocks["saddle"] = np.repeat(np.array(saddles).reshape(-1, n), 5, axis=0)
    for family in FAMILIES[1:]:
        moves = rng.normal(size=blocks[family].shape)
        blocks[family] = blocks[family] + 1e-2 * moves / np.linalg.norm(moves, axis=1, keepdims=True)
    labels = [family for family in FAMILIES for _ in range(len(blocks[family]))]
    return np.concatenate([blocks[family] for family in FAMILIES]), labels


def descend(psi: np.ndarray, table: np.ndarray, steps: int) -> np.ndarray:
    step = 1 / (3 * table.shape[1])
    for _ in range(steps):
        psi = psi - step * energy(psi, table)[1]
    return psi


def classify(psi: np.ndarray, table: np.ndarray) -> tuple[str, float]:
    """`character`, `minimizer` (a non-character local minimizer), `saddle`
    or `unconverged`, with the energy."""
    value, gradient = energy(psi, table)
    if value < 1e-8:
        return "character", float(value)
    if np.linalg.norm(gradient) > 1e-8:
        return "unconverged", float(value)
    least = np.linalg.eigvalsh(hessian(psi, table))[0]
    return ("minimizer" if least > -1e-8 else "saddle"), float(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--largest", type=int, default=30)
    parser.add_argument("--starts", type=int, default=400, help="starts per family")
    parser.add_argument("--newton", type=int, default=200, help="Newton starts")
    parser.add_argument("--steps", type=int, default=4000, help="gradient steps")
    arguments = parser.parse_args()
    print("n | n²/8 | runs | character | other ends | Newton points | lowest | index 0")
    for n in range(2, arguments.largest + 1):
        rng = np.random.default_rng(n)
        table = rows(n)
        saddles = newton_critical_points(n, table, arguments.newton, rng)
        psi, labels = starts(n, arguments.starts, saddles, rng)
        psi = descend(psi, table, arguments.steps)
        ends = [classify(point, table) for point in psi]
        other = sorted({(label, family, round(value, 4))
                        for (label, value), family in zip(ends, labels) if label != "character"})
        values = [energy(point, table)[0] for point in saddles]
        lowest = f"{min(values):.4f}" if values else "-"
        stable = sum(np.linalg.eigvalsh(hessian(point, table))[0] > -1e-8 for point in saddles)
        print(f"{n} | {n * n / 8:.3f} | {len(ends)} | "
              f"{sum(label == 'character' for label, _ in ends)} | {other or '-'} | "
              f"{len(saddles)} | {lowest} | {stable}", flush=True)


if __name__ == "__main__":
    main()
