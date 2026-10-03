"""Assemble complete paired calibration archives without averaging mechanisms.

Setting: Convexifying Transformers, Section 4. The prospective phase and
persistence criteria are explicit follow-up choices, not paper hyperparameters.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

from .stability_report import csv_rows, verify_archive, verify_csv, write_json


def optimizer_metadata(manifest):
    """Label optimizer comparisons while preserving historical raw-only reports."""
    optimizers = list(dict.fromkeys(row["config"]["optimizer"] for row in manifest["recipes"]))
    if optimizers == ["amsgradw"]:
        return {}
    labels = {"amsgradw": "raw AMSGradW", "adamw": "AdamW"}
    return {"optimizers": optimizers,
            "optimizer_label": " / ".join(labels.get(name, name) for name in optimizers)}


def comparison_rows(manifest, summaries):
    rows = []
    for recipe in manifest["recipes"]:
        summary = summaries[recipe["name"]]
        assessment = summary["assessment"]
        phase = assessment["legacy_phase_diagnostics"]
        event = assessment["long_confirmation"]
        config = summary["config"]
        rows.append({"name": recipe["name"], "changed_mechanism": recipe["changed_mechanism"],
            "learning_rate": config["learning_rate"], "batch_policy": config["batch_policy"],
            "model_seed": config["seed"], "data_seed": config["data_seed"], "steps": config["steps"],
            "parameters": summary["parameters"], "epochs_seen": summary["final"]["epochs_seen"],
            "training_seconds": summary["training_seconds"], "diagnostic_seconds": summary["diagnostic_seconds"],
            "wall_seconds": summary["wall_seconds"], "peak_cuda_allocated_bytes": summary["peak_cuda_allocated_bytes"],
            "final_train_accuracy": summary["final"]["train"]["accuracy"],
            "final_heldout_accuracy": summary["final"]["heldout"]["accuracy"],
            "final_heldout_answer_loss": summary["final"]["heldout"]["answer_loss"],
            "final_heldout_EOS_loss": summary["final"]["heldout"]["EOS_loss"],
            "legacy_plateau_then_generalization": phase["observed_plateau_then_generalization"],
            "restricted_epoch_error_double_descent": bool(phase["epoch_error_curve_before_generalization"]
                and phase["epoch_error_curve_before_generalization"]["full_error_double_descent"]),
            "long_confirmation_support": int(event is not None),
            "long_confirmation_onset": event["onset"] if event else None,
            "long_confirmation_step": event["confirmed"] if event else None,
            "long_confirmation_training_seconds": event["training_seconds"] if event else None,
            "long_confirmation_wall_seconds": event["wall_seconds"] if event else None,
            "tail_failures": len(assessment["tail_failures"]),
            "tail_minimum_heldout_accuracy": assessment["tail_minimum_heldout_accuracy"],
            "persistent_final_performance": assessment["persistent_final_performance"],
            "stable_grokking": assessment["stable_grokking"],
            "later_heldout_failures": summary["collapse_diagnostics"]["failures_after_confirmation"],
            **({"optimizer": config["optimizer"]} if optimizer_metadata(manifest) else {})})
    return rows


def comparison_summary(manifest, summaries):
    planned = {recipe["name"] for recipe in manifest["recipes"]}
    if set(summaries) != planned or not all(summary["complete_run"] for summary in summaries.values()):
        raise ValueError("Comparison requires every frozen complete run, including failures")
    for recipe in manifest["recipes"]:
        summary = summaries[recipe["name"]]
        if summary["config"] != recipe["config"] or not summary["assessment"]["complete_canonical_history"]:
            raise ValueError("Comparison recipe or complete history differs")
    rows = comparison_rows(manifest, summaries)
    eligible = [row["name"] for row in rows if row["stable_grokking"]]
    return {**optimizer_metadata(manifest),
            "complete_stage": True, "planned_runs": len(planned), "completed_runs": len(planned),
            "total_updates": sum(row["steps"] for row in rows),
            "total_canonical_observations": sum(summary["canonical_observations"] for summary in summaries.values()),
            "stable_grokking_calibration_recipes": eligible,
            "ready_to_freeze_independent_confirmation": bool(eligible),
            "independent_confirmation_complete": False,
            "criterion": manifest["criterion"], "environment": manifest["environment"], "rows": rows,
            "statistical_scope": "one_initialization_and_one_calibration_split_per_recipe; no_mean_SD_or_universal_speedup",
            "timing_scope": "first_20_observation_joint_target_confirmation; retain_tail_and_phase_failures; no_stopping_or_interpolation",
            "next_action": "select_a_passing_recipe_then_freeze_independent_confirmation" if eligible else
                           "preserve_negative_results_and_freeze_one_justified_mechanism_change_without_relaxing_criteria"}


def markdown_report(summary):
    optimizer = summary.get("optimizer_label", "raw AMSGradW")
    exposure = ("CSV retains actual exposure, losses, optimizer identities, phase flags, costs and memory."
                if "optimizer_label" in summary else
                "The sampling control consumes more examples at equal updates. CSV retains actual exposure, losses, phase flags, costs and memory.")
    lines = [f"# Complete {optimizer} stability calibration", "",
        "Source setting: *Convexifying Transformers*, arXiv:2211.11052v1, Section 4.",
        f"These GPTMini/{optimizer} experiments and stronger persistence targets are explicit adaptations.", "",
        f"All {summary['completed_runs']} frozen runs completed {summary['total_updates']:,} updates and "
        f"{summary['total_canonical_observations']:,} canonical observations. No failed target is omitted.", "",
        "| Recipe | Final train / held-out | Long confirmation, training s | Tail failures | Persistent tail | Stable grokking |",
        "| --- | ---: | ---: | ---: | --- | --- |"]
    for row in summary["rows"]:
        time = row["long_confirmation_training_seconds"]
        timing = f"{time:.2f}" if time is not None else "Not reached"
        lines.append(f"| {row['name']} | {100 * row['final_train_accuracy']:.4f}% / "
            f"{100 * row['final_heldout_accuracy']:.4f}% | {timing} | {row['tail_failures']} | "
            f"{row['persistent_final_performance']} | {row['stable_grokking']} |")
    lines.extend(["", "Each recipe is one initialization on one common calibration split. These are not independent confirmations.",
        "Timing is measured at the first 20-observation joint target confirmation; it does not imply subsequent persistence.",
        exposure,
        "Only recipes meeting the complete phase and final-tail criterion can enter a new independent confirmation plan.",
        "Neighbor/gradient associations do not establish a cause, an internal algorithm, or causality between grokking and double descent.", "",
        "Eligible calibration recipes: " + (", ".join(summary["stable_grokking_calibration_recipes"]) or "None") + ".", "",
        "Next action: `" + summary["next_action"] + "`.", ""])
    return "\n".join(lines)


def hashes(directory):
    return {str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(directory.rglob("*")) if path.is_file() and path != directory / "artifact-hashes.json"}


def assemble(paths, destination, *, render=True):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError("Comparison archive must be fresh")
    summaries, sources, manifest = {}, {}, None
    for path in map(Path, paths):
        summary = verify_archive(path)
        plan = json.loads((path / "plan.json").read_text())
        if manifest is not None and plan != manifest:
            raise ValueError("Calibration archives do not share the identical frozen plan")
        manifest = plan
        name = summary["name"]
        if name in summaries:
            raise ValueError("Duplicate calibration archive")
        summaries[name], sources[name] = summary, path
    if manifest is None:
        raise ValueError("Complete calibration archives are required")
    summary = comparison_summary(manifest, summaries)
    summary["plots_included"] = render
    directory = Path(__file__).parent
    summary["analysis_source_hashes"] = {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
        for name in ("stability_comparison.py", "stability_comparison_plots.py")}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=destination.name + "-", dir=destination.parent) as temporary:
        output = Path(temporary) / "archive"
        output.mkdir()
        for name, source in sources.items():
            shutil.copytree(source, output / "runs" / name)
        write_json(output / "plan.json", manifest)
        write_json(output / "summary.json", summary)
        csv_rows(output / "comparison.csv", summary["rows"])
        (output / "REPORT.md").write_text(markdown_report(summary))
        if render:
            from .stability_comparison_plots import render_comparison
            render_comparison(output, summary)
        write_json(output / "artifact-hashes.json", {"files": hashes(output)})
        verify_comparison(output)
        output.rename(destination)
    return summary


def verify_comparison(directory):
    directory = Path(directory)
    if hashes(directory) != json.loads((directory / "artifact-hashes.json").read_text())["files"]:
        raise ValueError("Comparison archive hashes differ")
    manifest = json.loads((directory / "plan.json").read_text())
    summaries = {}
    for recipe in manifest["recipes"]:
        path = directory / "runs" / recipe["name"]
        if json.loads((path / "plan.json").read_text()) != manifest:
            raise ValueError("Nested archive has a different frozen plan")
        summaries[recipe["name"]] = verify_archive(path)
    derived = comparison_summary(manifest, summaries)
    summary = json.loads((directory / "summary.json").read_text())
    if any(summary.get(key) != value for key, value in derived.items()):
        raise ValueError("Comparison differs from complete calibration outcomes")
    verify_csv(directory / "comparison.csv", derived["rows"])
    if (directory / "REPORT.md").read_text() != markdown_report(derived):
        raise ValueError("Comparison prose differs from measured outcomes")
    if summary["plots_included"] and any(not (directory / "plots" / f"calibration-comparison.{extension}").exists()
                                          for extension in ("png", "pdf")):
        raise ValueError("Comparison lacks declared curves")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)
    if args.verify:
        if len(args.paths) != 1:
            parser.error("Verification takes one comparison archive")
        summary = verify_comparison(args.paths[0])
    else:
        if args.archive is None:
            parser.error("Assembly requires --archive")
        summary = assemble(args.paths, args.archive)
    print(json.dumps({key: summary[key] for key in ("complete_stage", "completed_runs", "ready_to_freeze_independent_confirmation")}))


if __name__ == "__main__":
    main()
