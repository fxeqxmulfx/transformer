"""Portable all-case confirmation report; unsuccessful repeats cannot be omitted."""

import argparse
import json
from pathlib import Path
import shutil
import statistics
import tempfile

from .confirmation_layout import case_manifest, validate_plan
from .stability_comparison import comparison_rows, hashes
from .stability_report import csv_rows, verify_archive, verify_csv, write_json


def timing_summary(rows, field):
    values = [row[field] for row in rows if row[field] is not None]
    return {"support": len(values), "planned_runs": len(rows),
            "mean": statistics.mean(values) if values else None,
            "sample_SD": statistics.stdev(values) if len(values) > 1 else None,
            "scope": "descriptive_crossed_runs; sample_SD_is_not_a_confidence_interval; missing_events_excluded_with_support"}


def derive(plan, calibration, runs):
    validate_plan(plan, calibration)
    if set(runs) != {recipe["name"] for recipe in plan["recipes"]}:
        raise ValueError("Confirmation requires every frozen case, including failures")
    for recipe in plan["recipes"]:
        summary = runs[recipe["name"]]
        if not summary["complete_run"] or summary["config"] != recipe["config"]:
            raise ValueError("Confirmation run differs from the frozen recipe")
    rows = comparison_rows(plan, runs)
    successes = sum(row["stable_grokking"] for row in rows)
    return {"complete_confirmation_budget": True, "planned_runs": plan["planned_runs"],
            "task_prime": plan["recipes"][0]["config"]["prime"],
            "completed_runs": len(rows), "model_seeds": plan["model_seeds"], "data_seeds": plan["data_seeds"],
            "total_updates": sum(row["steps"] for row in rows),
            "total_canonical_observations": sum(s["canonical_observations"] for s in runs.values()),
            "stable_grokking_runs": successes, "repeatable_stable_benchmark": successes == len(rows),
            "architecture_comparison_ready": successes == len(rows), "rows": rows,
            "criterion": plan["criterion"], "environment": plan["environment"],
            "long_confirmation_training_seconds": timing_summary(rows, "long_confirmation_training_seconds"),
            "long_confirmation_wall_seconds": timing_summary(rows, "long_confirmation_wall_seconds"),
            "statistical_scope": plan["statistical_scope"],
            "timing_scope": "per_run_trainer_setup_and_synchronized_training; outer_verification_and_archive_costs_excluded",
            "next_action": "freeze_paired_architecture_comparison" if successes == len(rows) else
                           "retain_failed_repeats_and_continue_justified_calibration_without_relaxing_criteria"}


def markdown_report(summary):
    lines = ["# Independent raw AMSGradW / GPTMini confirmation", "",
        f"Setting: mod-{summary['task_prime']} division, adapting *Convexifying Transformers*, Section 4.",
        f"All {summary['completed_runs']} prospectively frozen repeats completed {summary['total_updates']:,} updates.",
        f"Stable grokking: {summary['stable_grokking_runs']}/{summary['planned_runs']}. "
        f"Repeatable benchmark: {summary['repeatable_stable_benchmark']}.", "",
        "| Initialization / data seed | Final held-out | Long confirmation, training s | Tail failures | Stable grokking |",
        "| --- | ---: | ---: | ---: | --- |"]
    for row in summary["rows"]:
        time = row["long_confirmation_training_seconds"]
        timing = f"{time:.2f}" if time is not None else "Not reached"
        lines.append(f"| {row['model_seed']} / {row['data_seed']} | {100 * row['final_heldout_accuracy']:.4f}% | "
                     f"{timing} | {row['tail_failures']} | {row['stable_grokking']} |")
    event = summary["long_confirmation_training_seconds"]
    lines.extend(["", f"Measured target timing support is {event['support']}/{event['planned_runs']}.",
        "Missing events remain empty. The CSV retains each outcome, phase flag, exposure, costs and memory.",
        "New initialization seeds are crossed with new data splits; these are not independent IID runs.",
        "Means and sample SD are descriptive. They do not establish a confidence interval or universal speedup.",
        "The unchanged calibration criterion applies to every complete trajectory, including all later failures.",
        "All repeats must pass before an architecture comparison is ready. A final rebound or first target does not replace persistence.",
        "The complete calibration evidence is included; checkpoints remain local and their fingerprints are retained.",
        "Finite scheduled observations do not identify an internal algorithm, a cause, or behavior beyond the budget.", "",
        "Next action: `" + summary["next_action"] + "`.", ""])
    return "\n".join(lines)


def read_runs(directory, plan, subdirectory):
    runs = {}
    for recipe in plan["recipes"]:
        path = directory / subdirectory / recipe["name"]
        if json.loads((path / "plan.json").read_text()) != case_manifest(plan, recipe):
            raise ValueError("Confirmation archive has a different frozen case plan")
        runs[recipe["name"]] = verify_archive(path)
    return runs


def assemble(directory, destination, *, render=True):
    directory, destination = Path(directory), Path(destination)
    if destination.exists():
        raise FileExistsError("Confirmation archive must be fresh")
    plan = json.loads((directory / "plan.json").read_text())
    runs = read_runs(directory, plan, "archives")
    summary = {**derive(plan, directory / "calibration", runs), "plots_included": render}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=destination.name + "-", dir=destination.parent) as temporary:
        output = Path(temporary) / "archive"
        output.mkdir()
        shutil.copytree(directory / "calibration", output / "calibration")
        for recipe in plan["recipes"]:
            name = recipe["name"]
            shutil.copytree(directory / "archives" / name, output / "runs" / name)
        write_json(output / "plan.json", plan)
        write_json(output / "summary.json", summary)
        csv_rows(output / "confirmation.csv", summary["rows"])
        (output / "REPORT.md").write_text(markdown_report(summary))
        if render:
            from .stability_comparison_plots import render_comparison
            render_comparison(output, summary, filename="confirmation-comparison", title=
                "Raw AMSGradW / unchanged GPTMini: independent frozen confirmation\n"
                "New initializations crossed with new splits; all complete budgets and failures")
        write_json(output / "artifact-hashes.json", {"files": hashes(output)})
        verify_confirmation(output)
        output.rename(destination)
    return summary


def verify_confirmation(directory):
    directory = Path(directory)
    if hashes(directory) != json.loads((directory / "artifact-hashes.json").read_text())["files"]:
        raise ValueError("Confirmation archive hashes differ")
    plan = json.loads((directory / "plan.json").read_text())
    runs = read_runs(directory, plan, "runs")
    derived = derive(plan, directory / "calibration", runs)
    summary = json.loads((directory / "summary.json").read_text())
    if any(summary.get(key) != value for key, value in derived.items()):
        raise ValueError("Confirmation summary differs from the full outcomes")
    verify_csv(directory / "confirmation.csv", derived["rows"])
    if (directory / "REPORT.md").read_text() != markdown_report(derived):
        raise ValueError("Confirmation prose differs from the full outcomes")
    if summary["plots_included"] and any(not (directory / "plots" / f"confirmation-comparison.{extension}").exists()
                                         for extension in ("png", "pdf")):
        raise ValueError("Confirmation archive lacks declared curves")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)
    if not args.verify and args.archive is None:
        parser.error("Assembly requires --archive")
    result = verify_confirmation(args.directory) if args.verify else assemble(args.directory, args.archive)
    print(json.dumps({key: result[key] for key in ("completed_runs", "stable_grokking_runs", "repeatable_stable_benchmark")}))


if __name__ == "__main__":
    main()
