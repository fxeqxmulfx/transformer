"""Evaluate the corrected FP4-SR scale policy against Quartet II baselines.

Source: arXiv:2601.22813v2, §3.1, §3.3, Algorithm 1, and Table 1.

The exact conditional expectation and MSE are computed from the two adjacent
grid points, so no rounding-coin Monte Carlo is needed for the error results.
Gaussian inputs are sampled to match Table 1. Because RHT is orthogonal and
the Gaussian law is rotation invariant, sampling the rotated vector directly
gives the same expected Euclidean MSE after inverse RHT.

The timing section measures these NumPy reference functions on the host CPU.
It is not a Blackwell kernel or GEMM benchmark.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np


FP4 = np.array([-6, -4, -3, -2, -1.5, -1, -0.5, 0,
                0.5, 1, 1.5, 2, 3, 4, 6], dtype=np.float64)
FP8 = np.array(sorted({0.0} | {
    float(m) * 2.0 ** e
    for e in range(-9, 9)
    for m in range(1, 16)
    if float(m) * 2.0 ** e <= 448.0
}), dtype=np.float64)
FP8_SR = np.concatenate((-FP8[:0:-1], FP8))
PAPER_S = 6.0 * (16.0 / 17.0) / 0.93


def neighbours(z: np.ndarray, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Saturating floor/ceiling, matching Section3_Grids.lean."""
    lower = np.clip(np.searchsorted(grid, z, side="right") - 1, 0, len(grid) - 1)
    upper = np.clip(np.searchsorted(grid, z, side="left"), 0, len(grid) - 1)
    return grid[lower], grid[upper]


def nearest(z: np.ndarray, grid: np.ndarray) -> np.ndarray:
    lo, hi = neighbours(z, grid)
    return np.where(z - lo <= hi - z, lo, hi)


def upward(z: np.ndarray, grid: np.ndarray) -> np.ndarray:
    return neighbours(z, grid)[1]


def sr_mean_var(z: np.ndarray, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Exact first moment and variance of stochastic grid rounding."""
    lo, hi = neighbours(z, grid)
    p = np.divide(z - lo, hi - lo, out=np.zeros_like(z), where=hi > lo)
    p = np.clip(p, 0.0, 1.0)
    mean = lo + p * (hi - lo)
    var = (1.0 - p) * (lo - mean) ** 2 + p * (hi - mean) ** 2
    return mean, var


def sr_draw(z: np.ndarray, grid: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    lo, hi = neighbours(z, grid)
    p = np.divide(z - lo, hi - lo, out=np.zeros_like(z), where=hi > lo)
    return np.where(rng.random(z.shape) < np.clip(p, 0.0, 1.0), hi, lo)


def block_max(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    groups = np.max(np.abs(y.reshape(-1, y.shape[-1] // 16, 16)), axis=-1)
    return np.max(groups, axis=-1, keepdims=True), groups


def sr_scales(y: np.ndarray, policy: str) -> tuple[np.ndarray, np.ndarray]:
    maximum, group_max = block_max(y)
    if policy == "paper_sr":
        tensor = np.where(maximum > 0, maximum / (6.0 * (16.0 / 17.0) * 448.0), 1.0)
        arg = np.divide(group_max, tensor * 6.0 * (16.0 / 17.0))
        group = nearest(arg, FP8)
    elif policy == "ceil_sr":
        tensor = np.where(maximum > 0, maximum / (6.0 * 448.0), 1.0 / 448.0)
        arg = group_max / (tensor * 6.0)
        group = np.where(group_max > 0, upward(arg, FP8), 1.0)
    elif policy == "unit_sr":
        tensor = np.where(maximum > 0, maximum / 6.0, 1.0)
        group = np.ones_like(group_max)
    else:
        raise ValueError(policy)
    return tensor, group


def sr_exact_stats(y: np.ndarray, policy: str) -> tuple[np.ndarray, np.ndarray, float]:
    tensor, group = sr_scales(y, policy)
    scale = np.repeat(group, 16, axis=-1) * tensor
    normalized = np.divide(y, scale, out=np.zeros_like(y), where=scale > 0)
    mean_fp4, var_fp4 = sr_mean_var(normalized, FP4)
    mean = mean_fp4 * scale
    variance = var_fp4 * scale ** 2
    mse = np.mean((mean - y) ** 2 + variance, axis=-1)
    bias_sq = np.mean((mean - y) ** 2, axis=-1)
    largest_arg = np.max(np.abs(normalized))
    return mse, bias_sq, float(largest_arg)


def sr_sample(y: np.ndarray, policy: str, rng: np.random.Generator) -> np.ndarray:
    tensor, group = sr_scales(y, policy)
    scale = np.repeat(group, 16, axis=-1) * tensor
    normalized = np.divide(y, scale, out=np.zeros_like(y), where=scale > 0)
    return sr_draw(normalized, FP4, rng) * scale


def eden_parts(y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    maximum, group_max = block_max(y)
    tensor = np.where(maximum > 0, maximum / (PAPER_S * 256.0), 1.0)
    group = nearest(group_max / (tensor * PAPER_S), FP8)
    scale = np.repeat(group, 16, axis=-1) * tensor
    normalized = np.divide(y, scale, out=np.zeros_like(y), where=scale > 0)
    fp4 = nearest(normalized, FP4)
    y_group = y.reshape(-1, y.shape[-1] // 16, 16)
    q_group = (fp4 * scale).reshape(y_group.shape)
    numerator = np.sum(y_group * y_group, axis=-1)
    denominator = np.sum(y_group * q_group, axis=-1)
    correction = np.divide(numerator, denominator, out=np.zeros_like(numerator),
                           where=denominator != 0)
    return tensor, fp4, group * correction


def eden_mean_var(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tensor, fp4, corrected_group = eden_parts(y)
    scale_mean, scale_var = sr_mean_var(corrected_group, FP8_SR)
    mean = fp4 * np.repeat(scale_mean, 16, axis=-1) * tensor
    variance = fp4 ** 2 * np.repeat(scale_var, 16, axis=-1) * tensor ** 2
    return mean, variance


def eden_exact_stats(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean, variance = eden_mean_var(y)
    return (np.mean((mean - y) ** 2 + variance, axis=-1),
            np.mean((mean - y) ** 2, axis=-1))


def eden_sample(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    tensor, fp4, corrected_group = eden_parts(y)
    sampled_group = sr_draw(corrected_group, FP8_SR, rng)
    return fp4 * np.repeat(sampled_group, 16, axis=-1) * tensor


def fwht(x: np.ndarray) -> np.ndarray:
    """Normalized Walsh-Hadamard transform with the ordering in Hadamard.lean."""
    out = np.array(x, dtype=np.float64, copy=True)
    width = out.shape[-1]
    step = 1
    while step < width:
        pairs = out.reshape(-1, width // (2 * step), 2, step)
        left = pairs[:, :, 0, :].copy()
        right = pairs[:, :, 1, :].copy()
        pairs[:, :, 0, :] = left + right
        pairs[:, :, 1, :] = left - right
        step *= 2
    return out / np.sqrt(width)


def witness_check() -> float:
    """Reproduce the Lean counterexample mean 101/102 at d=128."""
    x = np.zeros((1, 128))
    x[0, 0] = 1.0
    x[0, 1] = 0.1
    first = []
    for sign0 in (-1.0, 1.0):
        for sign1 in (-1.0, 1.0):
            signs = np.ones_like(x)
            signs[0, 0] = sign0
            signs[0, 1] = sign1
            y = fwht(x * signs)
            mean, _ = eden_mean_var(y)
            first.append(float((fwht(mean) * signs)[0, 0]))
    value = statistics.mean(first)
    if not np.isclose(value, 101.0 / 102.0, atol=1e-12, rtol=0):
        raise AssertionError(f"MS-EDEN witness mismatch: {value}")
    return value


def validate_reference() -> None:
    """Check grid moments and a tensor with zero and underflowing groups."""
    assert FP8[0] == 0.0 and FP8[-1] == 448.0 and 1.0 in FP8
    mean, variance = sr_mean_var(np.array([0.75]), FP4)
    assert np.isclose(mean[0], 0.75) and np.isclose(variance[0], 0.0625)
    y = np.zeros((1, 64))
    y[0, 0] = 1.0
    y[0, 16] = 1.0e-5
    y[0, 32] = 1.0e-7
    tensor, group = sr_scales(y, "ceil_sr")
    assert np.all(group > 0) and np.all(np.isin(group, FP8))
    normalized = y / (np.repeat(group, 16, axis=-1) * tensor)
    assert np.max(np.abs(normalized)) <= 6.0 + 1.0e-12
    assert sr_exact_stats(y, "ceil_sr")[1][0] < 1.0e-30
    assert sr_exact_stats(y, "paper_sr")[1][0] > 0.0
    witness_check()


def timed(fn, repeats: int) -> float:
    for _ in range(2):
        fn()
    runs = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        runs.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(runs)


def evaluate(samples: int, batch: int, repeats: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    names = ("paper_sr", "ceil_sr", "unit_sr", "ms_eden")
    totals = {name: [0.0, 0.0, 0.0, 0.0] for name in names}
    largest = {name: 0.0 for name in names[:3]}
    remaining = samples
    while remaining:
        y = rng.standard_normal((min(remaining, 4096), 128))
        remaining -= len(y)
        for name in names:
            if name == "ms_eden":
                errors, biases = eden_exact_stats(y)
            else:
                errors, biases, largest_arg = sr_exact_stats(y, name)
                largest[name] = max(largest[name], largest_arg)
            totals[name][0] += float(np.sum(errors))
            totals[name][1] += float(np.sum(errors ** 2))
            totals[name][2] += float(np.sum(biases))
            totals[name][3] += len(y)
    mse = {}
    for name in names:
        total, total_sq, bias_total, count = totals[name]
        mean = total / count
        sample_variance = max(0.0, (total_sq - total ** 2 / count) / (count - 1)) if count > 1 else 0.0
        mse[name] = {"mse": mean, "gaussian_standard_error": (sample_variance / count) ** 0.5,
                     "conditional_bias_sq": bias_total / count}
        if name in largest:
            mse[name]["largest_abs_fp4_argument"] = largest[name]

    speed_input = rng.standard_normal((batch, 128))
    speed_rng = np.random.default_rng(seed + 1)
    cpu_ms = {
        policy: timed(lambda p=policy: sr_sample(speed_input, p, speed_rng), repeats)
        for policy in ("paper_sr", "ceil_sr", "unit_sr")
    }
    cpu_ms["ms_eden"] = timed(lambda: eden_sample(speed_input, speed_rng), repeats)

    return {
        "source": "arXiv:2601.22813v2, sections 3.1 and 3.3, Algorithm 1, Table 1",
        "seed": seed, "gaussian_vectors": samples, "dimension": 128,
        "quantizer_batch": batch, "timing_repeats": repeats,
        "numpy": np.__version__, "cpu": platform.processor() or platform.machine(),
        "mse": mse, "cpu_numpy_median_ms_per_batch": cpu_ms,
        "paper_table_1_mse": {"paper_sr": 0.0235, "ms_eden": 0.0094},
        "rounding_coins_per_128_entries": {"paper_sr": 128, "ceil_sr": 128,
                                            "unit_sr": 128, "ms_eden": 8},
        "ms_eden_witness_mean_first_coordinate": witness_check(),
        "notes": [
            "MSE averages exact rounding-coin moments over sampled N(0,1) inputs.",
            "Conditional bias is after fixing a rotated input; it is not unconditional RHT bias.",
            "CPU NumPy timing excludes RHT, inverse RHT, GEMM, transfers, and GPU kernels.",
            "FP32 tensor scales are treated as real numbers, matching the Lean model.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=100_000)
    parser.add_argument("--batch", type=int, default=4096)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if args.samples < 1 or args.batch < 1 or args.repeats < 1:
        parser.error("samples, batch, and repeats must be positive")
    validate_reference()
    result = evaluate(args.samples, args.batch, args.repeats, args.seed)
    formatted = json.dumps(result, indent=2, sort_keys=True)
    if args.json:
        args.json.write_text(formatted + "\n")
    print(formatted)


if __name__ == "__main__":
    main()
