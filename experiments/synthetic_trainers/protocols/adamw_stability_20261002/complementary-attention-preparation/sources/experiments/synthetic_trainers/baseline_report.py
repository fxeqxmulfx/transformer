"""Aggregate saved baseline measurements without evaluating or selecting models."""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import shutil
import statistics

from .curves import curve_witness
from .runtime import write_json


def moments(values):
    values = [value for value in values if value is not None]
    return {"mean": statistics.mean(values), "std": statistics.stdev(values) if len(values) > 1 else 0,
            "support": len(values)} if values else None


def summarize(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    index = json.loads((directory / "runs.json").read_text())
    rows, grouped = [], defaultdict(list)
    for item in index:
        path = directory / item["result"]
        result = json.loads(path.read_text())
        history = [json.loads(line) for line in path.with_name("history.jsonl").read_text().splitlines()]
        final = history[-1]
        transition = result["study"]["generalization_transition"]
        config = result["provenance"]["training"]
        model = result["provenance"]["model"]
        phase = item["name"].split("/")[0]
        row = {"name": item["name"], "phase": phase, "variant": result["variant"],
               "width": model["width"], "layers": model["layers"], "seed": config["seed"],
               "data_seed": config["data_seed"], "noise": config["label_noise"],
               "parameters": result["parameters"], "steps": result["steps_completed"],
               "train_examples": config["train_examples"],
               "train_sequence_accuracy": final["train"]["sequence_accuracy"],
               "train_clean_sequence_accuracy": final["train_clean"]["sequence_accuracy"],
               "train_loss": final["train"]["example_loss"],
               "novel_validation_examples": transition["novel_validation_examples"],
               "novel_validation_accuracy": final["validation_novel"]["sequence_accuracy"] if final["validation_novel"] else None,
               "train_fit_step": transition["observed_train_fit_step"],
               "clean_train_fit_step": transition["clean_train_fit_step"],
               "id_generalization_step": transition["id_generalization_step"],
               "id_lag_steps": transition["id_lag_steps"],
               "transfer_step": transition["generalization_step"],
               "transfer_lag_steps": transition["lag_steps"],
               "delayed_id_candidate": transition["delayed_id_generalization_candidate"],
               "delayed_transfer_candidate": transition["delayed_transfer_candidate"],
               "test_id_accuracy": result["test_final"]["in_distribution"]["sequence_accuracy"],
               "test_id_loss": result["test_final"]["in_distribution"]["example_loss"],
               "selected_test_id_accuracy": result["test"]["in_distribution"]["sequence_accuracy"],
               "selected_test_id_loss": result["test"]["in_distribution"]["example_loss"],
               "training_seconds": result["training_seconds"], "wall_seconds": result["total_wall_seconds"],
               "peak_cuda_bytes": result["peak_cuda_bytes"],
               "membership_auc": result["study"]["final_checkpoint"]["loss_membership_auc"],
               "context_token_error_floor": result["study"]["causal_contexts"]["empirical_min_token_error"],
               "epoch_loss_witness": result["study"]["epoch_double_descent"]["validation_example_loss"],
               "epoch_error_witness": result["study"]["epoch_double_descent"]["validation_error"],
               "epoch_novel_error_witness": result["study"]["epoch_double_descent"].get("validation_novel_error"),
               "noise_fit": final.get("noise_fit"),
               "test_final": result["test_final"], "test_selected": result["test"],
               "source_result": item["result"]}
        compression = result["study"]["final_checkpoint"].get("train_compression")
        row["net_gain_bits"] = compression["net_gain_bits"] if compression else None
        row["compression"] = compression
        row["novel_validation_loss"] = final["validation_novel"]["example_loss"] if final["validation_novel"] else None
        row["has_epoch_loss_witness"] = row["epoch_loss_witness"] is not None
        row["has_epoch_error_witness"] = row["epoch_error_witness"] is not None
        row["has_epoch_novel_error_witness"] = row["epoch_novel_error_witness"] is not None
        row["noise_targets_changed"] = result["study"]["noise"]["changed_targets"]
        rows.append(row)
        grouped[phase, row["variant"], row["width"], row["noise"]].append(row)
    aggregates = []
    metrics = ("parameters", "train_sequence_accuracy", "train_clean_sequence_accuracy", "train_loss",
               "test_id_accuracy", "test_id_loss", "selected_test_id_accuracy", "selected_test_id_loss",
               "novel_validation_accuracy", "novel_validation_loss", "training_seconds", "wall_seconds",
               "membership_auc", "net_gain_bits", "id_lag_steps", "transfer_lag_steps")
    for (phase, variant, width, noise), group in grouped.items():
        aggregate = {"phase": phase, "variant": variant, "width": width, "noise": noise, "runs": len(group),
                     "model_seeds": [row["seed"] for row in group],
                     **{metric: moments([row[metric] for row in group]) for metric in metrics},
                     "train_fitted_runs": sum(row["train_sequence_accuracy"] > .99 for row in group),
                     "delayed_id_candidates": sum(row["delayed_id_candidate"] for row in group),
                     "delayed_transfer_candidates": sum(row["delayed_transfer_candidate"] for row in group),
                     "epoch_loss_witness_runs": sum(row["epoch_loss_witness"] is not None for row in group),
                     "epoch_error_witness_runs": sum(row["epoch_error_witness"] is not None for row in group),
                     "epoch_novel_error_witness_runs": sum(row["epoch_novel_error_witness"] is not None for row in group),
                     "test_final": {name: {key: moments([row["test_final"][name][key] for row in group])
                                            for key in ("sequence_accuracy", "token_accuracy", "example_loss")}
                                    for name in group[0]["test_final"]}}
        aggregates.append(aggregate)
    capacity = [row for row in aggregates if row["phase"] == "capacity"]
    capacity.sort(key=lambda row: row["parameters"]["mean"])
    capacity_witnesses = {metric: curve_witness(
        [(row["parameters"]["mean"], row[metric]["mean"]) for row in capacity], .02)
        for metric in ("test_id_loss",)}
    capacity_witnesses["test_id_error"] = curve_witness(
        [(row["parameters"]["mean"], 1 - row["test_id_accuracy"]["mean"]) for row in capacity], .02)
    capacity_seeds = []
    for seed in sorted({row["seed"] for row in rows if row["phase"] == "capacity"}):
        curve = sorted((row for row in rows if row["phase"] == "capacity" and row["seed"] == seed),
                       key=lambda row: row["parameters"])
        capacity_seeds.append({"seed": seed, "loss_witness": curve_witness(
            [(row["parameters"], row["test_id_loss"]) for row in curve], .02),
            "error_witness": curve_witness([(row["parameters"], 1 - row["test_id_accuracy"]) for row in curve], .02)})
    return {"manifest_status": manifest["status"], "planned_runs": manifest["planned_runs"],
            "completed_runs": len(rows), "hardware": {key: manifest[key] for key in
                ("gpu", "gpu_memory_bytes", "device", "torch", "cuda_runtime", "cpu_threads")},
            "training_seconds": sum(row["training_seconds"] for row in rows),
            "wall_seconds": sum(row["wall_seconds"] for row in rows),
            "curve_tolerance": .02, "capacity_mean_witnesses": capacity_witnesses,
            "capacity_seed_witnesses": capacity_seeds,
            "statistical_scope": "sample_SD_across_initialization_seeds; one_fixed_data_seed; finite_witnesses_not_significance",
            "runs": rows, "aggregates": aggregates}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--archive", type=Path, help="Keep compact measurements outside the ignored run directory")
    args = parser.parse_args(argv)
    summary = summarize(args.directory)
    write_json(args.directory / "summary.json", summary)
    flat = [{key: value for key, value in row.items() if not isinstance(value, (dict, list))}
            for row in summary["runs"]]
    with (args.directory / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted({key for row in flat for key in row}))
        writer.writeheader()
        writer.writerows(flat)
    if args.archive:
        archive(summary, args.directory, args.archive)
    print(json.dumps({key: summary[key] for key in ("completed_runs", "planned_runs", "training_seconds", "wall_seconds")}))


def archive(summary, directory, target):
    """Preserve scores, witnesses, hardware, recipes, and training source hashes."""
    directory, target = Path(directory), Path(target)
    if summary["manifest_status"] != "complete" or summary["completed_runs"] != summary["planned_runs"]:
        raise ValueError("Only a complete baseline can be archived")
    if target.exists() and any(target.iterdir()):
        raise FileExistsError("Archive directory must be empty")
    def score_summary(score):
        return {metric: value for metric, value in score.items()
                if metric in ("sequence_accuracy", "token_accuracy", "example_loss", "final_answer_accuracy")} if score else None

    compact = {**summary, "runs": []}
    for row in summary["runs"]:
        measured = dict(row)
        for key in ("test_final", "test_selected"):
            measured[key] = {name: score_summary(score) for name, score in row[key].items()}
        report = json.loads((directory / row["source_result"]).read_text())
        measured["split_fingerprints"] = report["split_fingerprints"]
        measured["source_hashes"] = report["provenance"]["source_hashes"]
        history = [json.loads(line) for line in (directory / row["source_result"]).with_name("history.jsonl").read_text().splitlines()]
        measured["history"] = [{"step": point["step"], "epochs_seen": point["epochs_seen"],
                                "training_seconds": point["training_seconds"],
                                **{key: score_summary(point[key]) for key in ("train", "train_clean", "validation", "validation_novel")},
                                "validation_ood": {name: score_summary(score) for name, score in point["validation_ood"].items()}}
                               for point in history]
        compact["runs"].append(measured)
    target.mkdir(parents=True, exist_ok=True)
    write_json(target / "measurements.json", compact)
    shutil.copyfile(directory / "manifest.json", target / "manifest.json")
    shutil.copyfile(directory / "metrics.csv", target / "metrics.csv")


if __name__ == "__main__":
    main()
