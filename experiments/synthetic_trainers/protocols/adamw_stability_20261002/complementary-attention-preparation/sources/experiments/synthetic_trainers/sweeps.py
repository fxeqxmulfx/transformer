"""Paired grids over model size, finite sample count, noise, and training time."""

from collections import defaultdict
import csv
from dataclasses import asdict, dataclass, replace
import itertools
import math
from pathlib import Path
import statistics

from .corpus import study_pool
from .curves import curve_witness, increase_witness
from .runtime import write_json
from .studies import validate_study
from .training import train_run


@dataclass(frozen=True)
class SweepConfig:
    widths: tuple[int, ...] = (32, 64, 128)
    layer_counts: tuple[int, ...] = (2,)
    sample_sizes: tuple[int, ...] = (128, 512, 2048)
    noise_rates: tuple[float, ...] = (0.0, 0.2)
    model_seeds: tuple[int, ...] = (0,)
    data_seeds: tuple[int, ...] = (0,)
    epochs: int | None = None

    def __post_init__(self):
        for values, minimum in ((self.widths, 1), (self.layer_counts, 1), (self.sample_sizes, 1),
                                (self.model_seeds, 0), (self.data_seeds, 0)):
            if not values or min(values) < minimum or len(set(values)) != len(values):
                raise ValueError("Sweep axes must be nonempty, distinct, and in range")
        if not self.noise_rates or len(set(self.noise_rates)) != len(self.noise_rates) or any(
                not math.isfinite(rate) or not 0 <= rate <= 1 for rate in self.noise_rates):
            raise ValueError("Noise rates must be distinct, finite, and in [0, 1]")
        if self.epochs is not None and self.epochs < 1:
            raise ValueError("Epoch budgets must be positive")


def run_name(width, layers, size, noise, data_seed, seed):
    return f"width-{width}-layers-{layers}/samples-{size}/noise-{noise}/data-{data_seed}-seed-{seed}"


def summarize(rows, tolerance=0.001):
    groups = defaultdict(list)
    for row in rows:
        key = (row["width"], row["layers"], row["train_examples"], row["label_noise"])
        groups[key].append(row)
    aggregates = []
    fields = ("parameters", "final_loss", "selected_loss", "final_error", "selected_error", "final_fit_value",
              "training_seconds", "net_gain_bits", "bits_per_parameter", "transfer_lag_steps", "transfer_lag_epochs",
              "transfer_lag_training_seconds")
    for (width, layers, size, noise), samples in sorted(groups.items()):
        item = {"width": width, "layers": layers, "train_examples": size, "label_noise": noise,
                "runs": len(samples), "fitted_runs": sum(row["final_fitted"] for row in samples),
                "transfer_confirmed_runs": sum(row.get("generalization_step") is not None for row in samples),
                "delayed_transfer_candidates": sum(row.get("delayed_transfer_candidate", False) for row in samples),
                "paired_replicates": [{"seed": row["seed"], "data_seed": row["data_seed"]} for row in samples]}
        for field in fields:
            values = [row[field] for row in samples if row.get(field) is not None]
            item[field] = {"mean": statistics.mean(values), "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                           "count": len(values)} if values else None
        aggregates.append(item)
    curves = []
    for axis in ("width", "layers", "train_examples"):
        grouped = defaultdict(list)
        fixed_keys = tuple(key for key in ("width", "layers", "train_examples", "label_noise") if key != axis)
        for item in aggregates:
            grouped[tuple(item[key] for key in fixed_keys)].append(item)
        for fixed, items in sorted(grouped.items()):
            items.sort(key=lambda item: item[axis])
            final_points = [(item[axis], item["final_loss"]["mean"]) for item in items]
            selected_points = [(item[axis], item["selected_loss"]["mean"]) for item in items]
            curves.append({"axis": axis, **dict(zip(fixed_keys, fixed)),
                           "points": items,
                           "final_loss_witness": curve_witness(final_points, tolerance),
                           "selected_loss_witness": curve_witness(selected_points, tolerance),
                           "final_error_witness": curve_witness([(item[axis], item["final_error"]["mean"]) for item in items], tolerance),
                           "selected_error_witness": curve_witness([(item[axis], item["selected_error"]["mean"]) for item in items], tolerance),
                           "more_data_hurts": increase_witness(final_points, tolerance) if axis == "train_examples" else None,
                           "scope": "sampled_mean_curve; variability_recorded; not_a_significance_test"})
    capacities = defaultdict(list)
    for item in aggregates:
        if item["net_gain_bits"] is not None:
            capacities[item["width"], item["layers"], item["label_noise"]].append(item)
    proxies = []
    for (width, layers, noise), items in sorted(capacities.items()):
        winner = max(items, key=lambda item: item["net_gain_bits"]["mean"])
        proxies.append({"width": width, "layers": layers, "label_noise": noise,
                        "maximum_mean_net_gain_bits": winner["net_gain_bits"]["mean"],
                        "train_examples_at_maximum": winner["train_examples"],
                        "bits_per_parameter": winner["bits_per_parameter"]["mean"],
                        "scope": "maximum_observed_final_likelihood_proxy; no_saturation_or_population_capacity_claim"})
    frontiers = defaultdict(list)
    for row in rows:
        frontiers[row["width"], row["layers"], row["label_noise"], row["data_seed"], row["seed"]].append(row)
    fit_frontiers = []
    for (width, layers, noise, data_seed, seed), samples in sorted(frontiers.items()):
        fitted = sorted(row["train_examples"] for row in samples if row["final_fitted"])
        unfitted = sorted(row["train_examples"] for row in samples if not row["final_fitted"])
        fit_frontiers.append({"width": width, "layers": layers, "label_noise": noise, "data_seed": data_seed, "seed": seed,
                              "fitted_sample_sizes": fitted, "unfitted_sample_sizes": unfitted,
                              "largest_observed_fitted_pool": max(fitted) if fitted else None,
                              "nonmonotone_fit": bool(fitted and any(size < max(fitted) for size in unfitted)),
                              "scope": "sampled_fit_frontier; not_EMC"})
    return {"aggregates": aggregates, "curves": curves, "capacity_proxies": proxies, "observed_fit_frontiers": fit_frontiers}


def run_sweep(spec, model_spec, config, sweep, output, *, model_factory=None, progress=None):
    if config.study == "standard":
        raise ValueError("Sweeps require memorization or double_descent profiles")
    for width, layers in itertools.product(sweep.widths, sweep.layer_counts):
        replace(model_spec, width=width, layers=layers)
    for noise in sweep.noise_rates:
        validate_study(spec, replace(config, label_noise=noise))
    if any(length <= spec.length for length in config.eval_lengths):
        raise ValueError("OOD evaluation lengths must exceed the training maximum")
    for length in config.eval_lengths:
        replace(spec, length=length, min_length=None)
    if config.target_metric == "final_answer_accuracy" and not spec.generative:
        raise ValueError("final_answer_accuracy requires generated answers")
    directory = Path(output)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError("Sweep directory is not empty; choose a fresh output directory")
    # Construct the largest pool once. Smaller pools are nested prefixes and
    # validation/test stay identical even with disjoint rejection sampling.
    pools = {seed: study_pool(spec, replace(config, data_seed=seed, train_examples=max(sweep.sample_sizes)))
             for seed in sweep.data_seeds}
    directory.mkdir(parents=True, exist_ok=True)
    planned = math.prod(len(axis) for axis in (sweep.widths, sweep.layer_counts, sweep.sample_sizes,
                                             sweep.noise_rates, sweep.data_seeds, sweep.model_seeds))
    manifest = {"task": asdict(spec), "model_template": asdict(model_spec), "training_template": asdict(config),
                "sweep": asdict(sweep), "planned_runs": planned}
    rows = []
    for width, layers, size, noise, data_seed, seed in itertools.product(
            sweep.widths, sweep.layer_counts, sweep.sample_sizes, sweep.noise_rates, sweep.data_seeds, sweep.model_seeds):
        pool = {**pools[data_seed], "train": replace(pools[data_seed]["train"], examples=pools[data_seed]["train"].examples[:size])}
        steps = config.steps if sweep.epochs is None else sweep.epochs * math.ceil(size / config.batch_size)
        run_config = replace(config, steps=steps, train_examples=size, label_noise=noise, data_seed=data_seed, seed=seed)
        output_run = directory / run_name(width, layers, size, noise, data_seed, seed)
        result = train_run(spec, replace(model_spec, width=width, layers=layers), run_config, output_run,
                           model_factory=model_factory, progress=progress, prepared_splits=pool)
        final = result["test_final"]["in_distribution"]
        selected = result["test"]["in_distribution"]
        compression = result["study"]["final_checkpoint"].get("train_compression")
        transition = result["study"]["generalization_transition"]
        rows.append({"width": width, "layers": layers, "train_examples": size, "label_noise": noise,
                     "data_seed": data_seed, "seed": seed, "parameters": result["parameters"],
                     "steps": steps, "epochs_seen": result["examples_seen"] / size,
                     "budget_mode": "updates" if sweep.epochs is None else "epochs",
                     "final_loss": final["example_loss"], "selected_loss": selected["example_loss"],
                     "final_error": 1 - final[config.target_metric], "selected_error": 1 - selected[config.target_metric],
                     "final_fit_value": result["study"]["interpolation"]["final_value"],
                     "final_fitted": result["study"]["interpolation"]["final_fitted"],
                     "training_seconds": result["training_seconds"],
                     "net_gain_bits": compression["net_gain_bits"] if compression else None,
                     "bits_per_parameter": compression["net_bits_per_parameter"] if compression else None,
                     "clean_train_fit_step": transition["clean_train_fit_step"],
                     "observed_train_fit_step": transition["observed_train_fit_step"],
                     "generalization_step": transition["generalization_step"],
                     "delayed_transfer_candidate": transition["delayed_transfer_candidate"],
                     "transfer_lag_steps": transition["lag_steps"], "transfer_lag_epochs": transition["lag_epochs"],
                     "transfer_lag_training_seconds": transition["lag_training_seconds"],
                     "result": str(output_run / "result.json")})
        write_json(directory / "sweep.json", {**manifest, "complete": len(rows) == planned,
                                               "runs": rows, **summarize(rows, config.curve_tolerance)})
        with (directory / "sweep.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return {**manifest, "complete": True, "runs": rows, **summarize(rows, config.curve_tolerance)}
