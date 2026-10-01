"""Reproduce the paper's Fashion-MNIST interpolation slices with fixed features."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import time

import numpy as np
import torch

from ..curves import curve_witness
from ..runtime import synchronize, write_json
from .fashion_data import load
from .random_features import features, fit_head, score


GRID = (100, 200, 300, 400, 500, 600, 700, 800, 900, 950, 975, 1000,
        1025, 1050, 1100, 1200, 1300, 1400, 1500, 1750, 2000)


def fingerprint(array):
    return hashlib.sha256(array.detach().cpu().numpy().tobytes()).hexdigest()


def aggregate(rows, tolerance=.02):
    groups = defaultdict(list)
    for row in rows:
        groups[row["samples"], row["width"]].append(row)
    means = []
    for (samples, width), group in sorted(groups.items()):
        item = {"samples": samples, "width": width, "runs": len(group),
                "interpolated_runs": sum(row["fit"]["interpolation_MSE"] < 1e-10 for row in group)}
        for split in ("train", "test"):
            item[split] = {metric: {"mean": statistics.mean(values := [row[split][metric] for row in group]),
                                   "std": statistics.stdev(values) if len(values) > 1 else 0}
                           for metric in ("error", "MSE")}
        means.append(item)
    curves = []
    for axis, fixed, fixed_value in (("samples", "width", 1000), ("width", "samples", 1000)):
        points = sorted((row for row in means if row[fixed] == fixed_value), key=lambda row: row[axis])
        by_seed = []
        for seed in sorted({row["seed"] for row in rows}):
            measured = sorted((row for row in rows if row["seed"] == seed and row[fixed] == fixed_value), key=lambda row: row[axis])
            by_seed.append({"seed": seed, "error_witness": curve_witness([(row[axis], row["test"]["error"]) for row in measured], tolerance),
                            "MSE_witness": curve_witness([(row[axis], row["test"]["MSE"]) for row in measured], tolerance)})
        curves.append({"axis": axis, "fixed": {fixed: fixed_value}, "points": points, "by_seed": by_seed,
                       "mean_error_witness": curve_witness([(row[axis], row["test"]["error"]["mean"]) for row in points], tolerance),
                       "mean_MSE_witness": curve_witness([(row[axis], row["test"]["MSE"]["mean"]) for row in points], tolerance)})
    return {"aggregates": means, "curves": curves}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("experiments/runs/paper_data/fashion-mnist"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--grid", type=int, nargs="+", default=list(GRID))
    args = parser.parse_args(argv)
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output directory must be empty")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    if min(args.grid) < 1 or max(args.grid) > 60000 or len(set(args.grid)) != len(args.grid):
        parser.error("Invalid grid of sample counts / widths")
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    train_images, train_labels, test_images, test_labels, dataset = load(args.data)
    shapes = sorted({(size, 1000) for size in args.grid} | {(1000, width) for width in args.grid})
    source_dir = Path(__file__).parent
    plan = {"status": "running", "source": "papers/arXiv-1912.02292v1/paper.txt, Appendix C, Figures 14–15",
            "scope": "random_feature_case_study_reproduction; not_CNN_or_translation_reproduction",
            "architecture": "frozen_complex_exp(-i*x)_features; zero_initialized_complex_linear_MSE_head",
            "feature_weight_variance": "1/feature_width", "solver": "zero_initialization_gradient_flow_limit_by_QR",
            "dtype": "float64/complex128", "pixel_normalization": "uint8/255",
            "prediction_rule": "argmax_real_part; full_complex_MSE_reported",
            "undisclosed_paper_details": ["pixel_normalization", "exact_data/feature_seeds", "finite_gradient_flow_time", "complex_prediction_classification_rule"],
            "grid": args.grid, "seeds": args.seeds, "planned_runs": len(shapes) * len(args.seeds),
            "data_sampling": "nested_uniform_without_replacement; independent_official_test_set",
            "curve_tolerance": .02, "dataset": dataset, "torch": torch.__version__, "device": args.device,
            "gpu": torch.cuda.get_device_name(args.device) if args.device.startswith("cuda") else None,
            "source_hashes": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(source_dir.glob("*.py"))}}
    write_json(args.output / "plan.json", plan)
    rows, pools = [], []
    start = time.perf_counter()
    maximum = max(1000, max(args.grid))
    test_x = torch.tensor(test_images, dtype=torch.float64, device=args.device) / 255
    test_y = torch.tensor(test_labels.astype(np.int64), device=args.device)
    for seed in args.seeds:
        indices = np.random.RandomState(seed).permutation(len(train_images))[:maximum]
        train_x = torch.tensor(train_images[indices], dtype=torch.float64, device=args.device) / 255
        train_y = torch.tensor(train_labels[indices].astype(np.int64), device=args.device)
        gaussian = torch.randn((train_x.shape[1], maximum), dtype=torch.float64,
                               generator=torch.Generator().manual_seed(seed + 1000)).to(args.device)
        pools.append({"seed": seed, "training_indices": indices.tolist(), "gaussian_fingerprint": fingerprint(gaussian)})
        for width in sorted({width for _, width in shapes}):
            all_features = features(train_x, gaussian, width)
            test_features = features(test_x, gaussian, width)
            for size, _ in (shape for shape in shapes if shape[1] == width):
                synchronize(args.device)
                started = time.perf_counter()
                head, fit = fit_head(all_features[:size], train_y[:size])
                synchronize(args.device)
                fitting_seconds = time.perf_counter() - started
                row = {"samples": size, "width": width, "seed": seed, "fit": fit,
                       "train": score(all_features[:size], head, train_y[:size]),
                       "test": score(test_features, head, test_y), "fitting_seconds": fitting_seconds}
                rows.append(row)
                print(json.dumps(row), flush=True)
                write_json(args.output / "measurements.json", {"plan": plan, "completed_runs": len(rows), "runs": rows,
                    "pools": pools, **aggregate(rows)})
                torch.save(head.cpu(), args.output / f"head-seed-{seed}-N-{size}-d-{width}.pt")
    plan.update(status="complete", elapsed_seconds=time.perf_counter() - start)
    write_json(args.output / "plan.json", plan)
    write_json(args.output / "measurements.json", {"plan": plan, "completed_runs": len(rows), "runs": rows,
               "pools": pools, **aggregate(rows)})


if __name__ == "__main__":
    main()
