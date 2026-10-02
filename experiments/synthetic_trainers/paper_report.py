"""Archive complete paper experiments and describe measured interpolation peaks."""

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import shutil
import statistics

from .runtime import write_json
from .paper_phases import diagnose


def moments(values):
    return {"mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0, "support": len(values)} if values else None


def peak_summary(points, fixed, margin=.02):
    """Describe the actual global peak, rather than the first finite witness."""
    points = sorted(points, key=lambda row: row["x"])
    if len(points) < 4:
        return None
    peak = max(points, key=lambda row: row["error"])
    before = [row for row in points if row["x"] < peak["x"]]
    if not before:
        return None
    minimum = min(before, key=lambda row: row["error"])
    initial, tail = points[0], points[-1]
    shape = (initial["x"] < minimum["x"] < peak["x"] < tail["x"]
             and initial["error"] > minimum["error"] + margin
             and peak["error"] > minimum["error"] + margin
             and peak["error"] > tail["error"] + margin)
    return {"initial": initial, "first_minimum": minimum, "peak": peak, "final": tail,
            "full_error_double_descent": shape, "margin": margin,
            "peak_on_interpolation_boundary": peak["x"] == fixed,
            "peak_rise": peak["error"] - minimum["error"],
            "second_descent": peak["error"] - tail["error"],
            "second_descent_beats_first_minimum": tail["error"] < minimum["error"] - margin,
            "scope": "posthoc_description_of_all_frozen_grid_points; not_significance_or_hyperparameter_selection"}


def rff_summary(report):
    summaries = []
    for curve in report["curves"]:
        axis = curve["axis"]
        fixed_key, fixed_value = next(iter(curve["fixed"].items()))
        means = [{"x": row[axis], "error": row["test"]["error"]["mean"],
                  "std": row["test"]["error"]["std"], "runs": row["runs"],
                  "interpolated_runs": row["interpolated_runs"]} for row in curve["points"]]
        by_seed = []
        for seed in report["plan"]["seeds"]:
            rows = [row for row in report["runs"] if row["seed"] == seed and row[fixed_key] == fixed_value]
            points = [{"x": row[axis], "error": row["test"]["error"],
                       "train_MSE": row["fit"]["interpolation_MSE"]} for row in rows]
            by_seed.append({"seed": seed, "error_curve": peak_summary(points, fixed_value),
                            "interpolated_points": sorted(row[axis] for row in rows if row["fit"]["interpolation_MSE"] < 1e-10)})
        summaries.append({"axis": axis, "fixed": curve["fixed"],
                          "mean_error_curve": peak_summary(means, fixed_value), "by_seed": by_seed,
                          "repeated_full_error_double_descent": sum(bool(row["error_curve"] and row["error_curve"]["full_error_double_descent"]) for row in by_seed),
                          "peak_on_boundary_runs": sum(bool(row["error_curve"] and row["error_curve"]["peak_on_interpolation_boundary"]) for row in by_seed),
                          "mean_MSE_four_point_witness": curve["mean_MSE_witness"]})
    return {"complete": report["plan"]["status"] == "complete" and report["completed_runs"] == report["plan"]["planned_runs"],
            "completed_runs": report["completed_runs"], "planned_runs": report["plan"]["planned_runs"],
            "elapsed_seconds": report["plan"].get("elapsed_seconds"),
            "fitting_seconds": sum(row["fitting_seconds"] for row in report["runs"]),
            "maximum_interpolation_MSE_at_or_above_width": max(row["fit"]["interpolation_MSE"] for row in report["runs"] if row["width"] >= row["samples"]),
            "maximum_normal_equation_residual": max(row["fit"]["normal_equation_residual"] for row in report["runs"]),
            "source": report["plan"]["source"], "undisclosed_paper_details": report["plan"]["undisclosed_paper_details"],
            "statistical_scope": "three_paired_independent_data_and_feature_seeds; sample_SD_not_confidence_intervals",
            "curves": summaries}


def modular_summary(reports, planned):
    groups = defaultdict(list)
    for report in reports:
        config = report["plan"]["config"]
        groups[config["model"], config["optimizer"]].append(report)
    aggregates = []
    for (model, optimizer), group in sorted(groups.items()):
        phases = [diagnose(row) for row in group]
        aggregates.append({"model": model, "optimizer": optimizer, "runs": len(group),
            "confirmed_delayed_runs": sum(row["transition"]["delayed_generalization"] for row in group),
            "final_heldout_target_runs": sum(row["final"]["heldout"]["accuracy"] >= row["plan"]["config"]["target"] for row in group),
            "lag_steps": moments([row["transition"]["lag_steps"] for row in group if row["transition"]["lag_steps"] is not None]),
            "train_fit_step": moments([row["transition"]["train_fit_step"] for row in group if row["transition"]["train_fit_step"] is not None]),
            "heldout_onset_step": moments([row["transition"]["heldout_onset_step"] for row in group if row["transition"]["heldout_onset_step"] is not None]),
            "heldout_target_onset_training_seconds": moments([phase["time_to_sustained_heldout_target"]["onset"]["training_seconds"] for phase in phases if phase["time_to_sustained_heldout_target"] is not None]),
            "heldout_target_confirmed_training_seconds": moments([phase["time_to_sustained_heldout_target"]["confirmed"]["training_seconds"] for phase in phases if phase["time_to_sustained_heldout_target"] is not None]),
            "heldout_target_onset_wall_seconds": moments([phase["time_to_sustained_heldout_target"]["onset"]["wall_seconds"] for phase in phases if phase["time_to_sustained_heldout_target"] is not None]),
            "heldout_target_confirmed_wall_seconds": moments([phase["time_to_sustained_heldout_target"]["confirmed"]["wall_seconds"] for phase in phases if phase["time_to_sustained_heldout_target"] is not None]),
            "training_seconds": moments([row["training_seconds"] for row in group]),
            "heldout_accuracy": moments([row["final"]["heldout"]["accuracy"] for row in group]),
            "heldout_loss": moments([row["final"]["heldout"]["loss"] for row in group]),
            "plateau_then_generalization_runs": sum(row["observed_plateau_then_generalization"] for row in phases),
            "epoch_error_double_descent_runs": sum(bool(row["epoch_error_curve_before_generalization"] and row["epoch_error_curve_before_generalization"]["full_error_double_descent"]) for row in phases),
            "both_epoch_error_double_descent_and_grokking_runs": sum(row["both_epoch_error_double_descent_and_grokking"] for row in phases),
            "lag_after_sustained_train_fit": moments([row["lag_after_sustained_train_fit"] for row in phases if row["lag_after_sustained_train_fit"] is not None]),
            "phase_diagnostics": [{"seed": report["plan"]["config"]["seed"], **phase} for report, phase in zip(group, phases)]})
    return {"complete": len(reports) == planned and all(row["plan"]["status"] == "complete" for row in reports),
            "completed_runs": len(reports), "planned_runs": planned, "aggregates": aggregates,
            "training_seconds": sum(row["training_seconds"] for row in reports),
            "statistical_scope": "initialization_seeds_on_one_exhaustive_two_way_split; no_length_transfer_or_causal_claim",
            "runs": reports}


def load_modular(directory):
    path = directory / "campaign.json"
    if path.exists():
        campaign = json.loads(path.read_text())
        reports = [json.loads(Path(row["result"]).read_text()) for row in campaign["runs"]]
        return modular_summary(reports, campaign["planned_runs"])
    report = json.loads((directory / "measurements.json").read_text())
    if "runs" in report and "planned_runs" in report:
        return modular_summary(report["runs"], report["planned_runs"])
    return modular_summary([report], 1)


def export_csv(kind, report, path):
    rows = []
    if kind == "rff":
        for row in report["runs"]:
            rows.append({**{key: row[key] for key in ("samples", "width", "seed", "fitting_seconds")},
                         **{f"{split}_{key}": value for split in ("train", "test", "fit") for key, value in row[split].items()}})
    else:
        for row in report["runs"]:
            config = row["plan"]["config"]
            phase = diagnose(row)
            heldout_times = phase["time_to_sustained_heldout_target"]
            rows.append({**{key: config[key] for key in ("model", "optimizer", "seed", "width", "layers", "steps")},
                         "parameters": row["plan"]["parameters"], "training_seconds": row["training_seconds"],
                         "sustained_train_fit_step": phase["sustained_train_fit"]["onset"] if phase["sustained_train_fit"] else None,
                         "lag_after_sustained_train_fit": phase["lag_after_sustained_train_fit"],
                         "plateau_then_generalization": phase["observed_plateau_then_generalization"],
                         "epoch_error_double_descent": bool(phase["epoch_error_curve_before_generalization"] and phase["epoch_error_curve_before_generalization"]["full_error_double_descent"]),
                         "both_epoch_error_double_descent_and_grokking": phase["both_epoch_error_double_descent_and_grokking"],
                         "heldout_target_observation_fraction": phase["fraction_observations_at_target_after_confirmation"],
                         **{f"heldout_target_{event}_{clock}": heldout_times[event][clock] if heldout_times else None
                            for event in ("onset", "confirmed") for clock in ("training_seconds", "wall_seconds")},
                         **{key: row["transition"][key] for key in ("train_fit_step", "heldout_onset_step", "lag_steps", "delayed_generalization")},
                         **{f"final_{split}_{key}": value for split in ("train", "heldout") for key, value in row["final"][split].items()}})
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted({key for row in rows for key in row}), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def save(kind, directory, archive=None):
    directory = Path(directory)
    report = json.loads((directory / "measurements.json").read_text()) if kind == "rff" else load_modular(directory)
    summary = rff_summary(report) if kind == "rff" else {key: value for key, value in report.items() if key != "runs"}
    directory_source = Path(__file__).parent
    summary["analysis_source_hashes"] = {name: hashlib.sha256((directory_source / name).read_bytes()).hexdigest()
                                         for name in ("paper_report.py", "paper_phases.py", "paper_plots.py")}
    if kind == "modular" and (directory / "campaign.json").exists():
        write_json(directory / "measurements.json", report)
    write_json(directory / "summary.json", summary)
    export_csv(kind, report, directory / "metrics.csv")
    if archive is not None:
        archive = Path(archive)
        if not summary["complete"]:
            raise ValueError("Only complete paper experiments can be archived")
        if archive.exists() and any(archive.iterdir()):
            raise FileExistsError("Archive must be fresh")
        archive.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(directory / "plan.json", archive / "plan.json")
        for name in ("summary.json", "metrics.csv"):
            shutil.copyfile(directory / name, archive / name)
        write_json(archive / "measurements.json", report)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("rff", "modular"))
    parser.add_argument("directory", type=Path)
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args(argv)
    summary = save(args.kind, args.directory, args.archive)
    print(json.dumps({key: value for key, value in summary.items() if key not in ("curves", "aggregates")}))


if __name__ == "__main__":
    main()
