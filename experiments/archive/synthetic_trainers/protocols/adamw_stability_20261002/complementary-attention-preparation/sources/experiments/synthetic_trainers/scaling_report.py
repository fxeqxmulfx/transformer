"""Preserve measured scaling results, paired pools, and finite curve witnesses."""

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


def score_summary(score):
    fields = ("sequence_accuracy", "token_accuracy", "example_loss", "final_answer_accuracy")
    return {key: score[key] for key in fields if key in score} if score else None


def measured_run(directory, phase, item):
    path = Path(item["result"])
    result = json.loads(path.read_text())
    history = [json.loads(line) for line in path.with_name("history.jsonl").read_text().splitlines()]
    final, study, provenance = history[-1], result["study"], result["provenance"]
    config, model = provenance["training"], provenance["model"]
    transition = study["generalization_transition"]
    row = {"phase": phase, "variant": result["variant"],
           "source_result": str(path.resolve().relative_to(Path(directory).resolve())),
           **{key: model[key] for key in ("width", "layers", "heads")},
           **{key: config[key] for key in ("seed", "data_seed", "train_examples", "label_noise")},
           "parameters": result["parameters"], "steps": result["steps_completed"],
           "epochs_seen": result["examples_seen"] / config["train_examples"],
           "training_seconds": result["training_seconds"], "wall_seconds": result["total_wall_seconds"],
           "peak_cuda_bytes": result["peak_cuda_bytes"],
           "train_accuracy": final["train"]["sequence_accuracy"],
           "train_clean_accuracy": final["train_clean"]["sequence_accuracy"],
           "train_loss": final["train"]["example_loss"],
           "final_fitted": study["interpolation"]["final_fitted"],
           "final_fit_value": study["interpolation"]["final_value"],
           "fit_epsilon": study["interpolation"]["epsilon"],
           "novel_validation_examples": transition["novel_validation_examples"],
           "novel_validation_accuracy": final["validation_novel"]["sequence_accuracy"] if final["validation_novel"] else None,
           "novel_validation_loss": final["validation_novel"]["example_loss"] if final["validation_novel"] else None,
           "id_test_accuracy": result["test_final"]["in_distribution"]["sequence_accuracy"],
           "id_test_loss": result["test_final"]["in_distribution"]["example_loss"],
           "selected_id_test_accuracy": result["test"]["in_distribution"]["sequence_accuracy"],
           "selected_id_test_loss": result["test"]["in_distribution"]["example_loss"],
           "realized_noise": study["noise"]["realized_rate"],
           "noise_targets_changed": study["noise"]["changed_targets"],
           "generalization_transition": transition,
           "epoch_double_descent": study["epoch_double_descent"],
           "corpus": study["corpus"], "split_fingerprints": result["split_fingerprints"],
           "provenance": provenance,
           "test_final": {name: score_summary(score) for name, score in result["test_final"].items()},
           "test_selected": {name: score_summary(score) for name, score in result["test"].items()}}
    row["history"] = [{"step": point["step"], "epochs_seen": point["epochs_seen"],
                        "training_seconds": point["training_seconds"],
                        **{key: score_summary(point[key]) for key in ("train", "train_clean", "validation", "validation_novel")},
                        **{key: {name: score_summary(score) for name, score in point[key].items()}
                           for key in ("validation_ood", "validation_ood_novel")}}
                       for point in history]
    return row


def aggregate_runs(rows):
    """Keep sample count and depth in grouping keys; do not mix ablations."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in ("phase", "width", "layers", "train_examples", "label_noise"))].append(row)
    metrics = ("parameters", "train_accuracy", "train_clean_accuracy", "train_loss", "final_fit_value",
               "novel_validation_accuracy", "novel_validation_loss", "id_test_accuracy", "id_test_loss",
               "selected_id_test_accuracy", "selected_id_test_loss", "realized_noise", "training_seconds", "wall_seconds")
    aggregates = []
    for key, group in sorted(grouped.items()):
        item = {**dict(zip(("phase", "width", "layers", "train_examples", "label_noise"), key)),
                "runs": len(group), "fitted_runs": sum(row["final_fitted"] for row in group),
                "model_seeds": sorted({row["seed"] for row in group}),
                "data_seeds": sorted({row["data_seed"] for row in group}),
                **{metric: moments([row[metric] for row in group]) for metric in metrics},
                "delayed_id_candidates": sum(row["generalization_transition"]["delayed_id_generalization_candidate"] for row in group),
                "delayed_transfer_candidates": sum(row["generalization_transition"]["delayed_transfer_candidate"] for row in group)}
        item["test_final"] = {name: {metric: moments([row["test_final"][name][metric] for row in group])
                                    for metric in ("sequence_accuracy", "token_accuracy", "example_loss")}
                              for name in group[0]["test_final"]}
        aggregates.append(item)
    return aggregates


def width_witnesses(rows, aggregates, tolerance):
    width_rows = [row for row in rows if row["phase"] == "dd-widths"]
    width_means = sorted((row for row in aggregates if row["phase"] == "dd-widths"), key=lambda row: row["width"])
    # This phase has one N and depth; refuse accidental mixing if reused elsewhere.
    if len({(row["layers"], row["train_examples"], row["label_noise"]) for row in width_means}) > 1:
        raise ValueError("Width witnesses require a fixed depth, sample count, and noise")
    means = {"loss": curve_witness([(row["width"], row["id_test_loss"]["mean"]) for row in width_means], tolerance),
             "error": curve_witness([(row["width"], 1 - row["id_test_accuracy"]["mean"]) for row in width_means], tolerance)}
    seeds = []
    for data_seed, seed in sorted({(row["data_seed"], row["seed"]) for row in width_rows}):
        points = sorted((row for row in width_rows if row["data_seed"] == data_seed and row["seed"] == seed), key=lambda row: row["width"])
        seeds.append({"data_seed": data_seed, "seed": seed,
                      "loss": curve_witness([(row["width"], row["id_test_loss"]) for row in points], tolerance),
                      "error": curve_witness([(row["width"], 1 - row["id_test_accuracy"]) for row in points], tolerance),
                      "fit_by_width": {str(row["width"]): row["final_fitted"] for row in points}})
    return {"mean": means, "by_seed": seeds, "tolerance": tolerance,
            "scope": "four_ordered_measured_points; unfitted_runs_retained; not_significance_or_causality"}


def paired_pools(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["phase"], row["train_examples"], row["label_noise"], row["data_seed"]].append(row)
    return [{"phase": key[0], "train_examples": key[1], "label_noise": key[2], "data_seed": key[3],
             "runs": len(group),
             "all_split_fingerprints_identical": len({json.dumps(row["split_fingerprints"], sort_keys=True) for row in group}) == 1}
            for key, group in sorted(grouped.items())]


def summarize(directory):
    directory = Path(directory)
    plan = json.loads((directory / "plan.json").read_text())
    phases = {"all": ("dd-calibration", "dd-widths", "copy-scaling"),
              "calibration": ("dd-calibration",), "copy": ("copy-scaling",)}[plan["phase"]]
    rows, sweeps = [], {}
    for phase in phases:
        path = directory / phase / "sweep.json"
        if path.exists():
            sweep = json.loads(path.read_text())
            sweeps[phase] = {key: value for key, value in sweep.items() if key != "runs"}
            rows.extend(measured_run(directory, phase, item) for item in sweep["runs"])
    aggregates = aggregate_runs(rows)
    planned = sum(sweep["planned_runs"] for sweep in sweeps.values()) if len(sweeps) == len(phases) else None
    return {"status": plan["status"], "plan": plan, "planned_runs": planned, "completed_runs": len(rows),
            "complete": plan["status"] == "complete" and len(sweeps) == len(phases) and all(sweep["complete"] for sweep in sweeps.values()),
            "training_seconds": sum(row["training_seconds"] for row in rows),
            "wall_seconds": sum(row["wall_seconds"] for row in rows),
            "peak_cuda_bytes": max((row["peak_cuda_bytes"] or 0 for row in rows), default=0),
            "statistical_scope": "sample_SD_across_initialization_seeds; one_data_seed; copy_ablation_one_seed; finite_fit_not_EMC",
            "width_witnesses": width_witnesses(rows, aggregates, plan["dd_training"]["curve_tolerance"]),
            "paired_pools": paired_pools(rows), "sweeps": sweeps, "runs": rows, "aggregates": aggregates}


def save(summary, directory, target=None):
    directory = Path(directory)
    write_json(directory / "measurements.json", summary)
    flat = [{key: value for key, value in row.items() if not isinstance(value, (dict, list))} for row in summary["runs"]]
    with (directory / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted({key for row in flat for key in row}))
        writer.writeheader()
        writer.writerows(flat)
    if target is not None:
        target = Path(target)
        if not summary["complete"] or summary["completed_runs"] != summary["planned_runs"]:
            raise ValueError("Only a complete scaling experiment can be archived")
        if target.exists() and any(target.iterdir()):
            raise FileExistsError("Archive directory must be empty")
        target.mkdir(parents=True, exist_ok=True)
        for name in ("plan.json", "measurements.json", "metrics.csv"):
            shutil.copyfile(directory / name, target / name)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args(argv)
    summary = summarize(args.directory)
    save(summary, args.directory, args.archive)
    print(json.dumps({key: summary[key] for key in ("complete", "completed_runs", "planned_runs", "training_seconds", "wall_seconds")}))


if __name__ == "__main__":
    main()
