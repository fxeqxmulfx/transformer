"""Export validation learning times without loading models or using the GPU."""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

from .milestones import THRESHOLDS, summarize_epochs


def read_runs(root, attention, lengths):
    runs = []
    for path in sorted(Path(root).glob("n*-d*-lr*/epochs.jsonl")):
        if int(path.parent.name.split("-", 1)[0][1:]) not in lengths:
            continue
        raw = path.read_bytes()
        if not raw.endswith(b"\n"):
            raise ValueError(f"An epoch log has an incomplete final line: {path}")
        rows = [json.loads(line) for line in raw.splitlines()]
        if not rows or rows[0]["length"] not in lengths:
            continue
        metadata = {key: rows[0][key] for key in ("length", "width", "learning_rate")}
        if any(any(row[key] != value for key, value in metadata.items()) for row in rows):
            raise ValueError(f"An epoch log mixes configurations: {path}")
        runs.append({"attention": attention, **metadata, "directory": str(path.parent),
                     "epoch_log_sha256": hashlib.sha256(raw).hexdigest(),
                     "completed": (path.parent / "result.json").exists(),
                     **summarize_epochs(rows)})
    return runs


def build_report(softmax, sparsemax, lengths=(64, 128, 256), selection="stable99"):
    if selection not in ("stable99", "first99", "best"):
        raise ValueError(f"Unknown validation selection policy: {selection}")
    roots = {"softmax": Path(softmax), "sparsemax": Path(sparsemax)}
    configs = {}
    provenance = {}
    all_runs = []
    for attention, root in roots.items():
        metadata = json.loads((root / "config.json").read_text())
        configs[attention] = metadata.get("config", metadata)
        provenance[attention] = {
            "directory": str(root),
            "environment": json.loads((root / "environment.json").read_text())
                if (root / "environment.json").exists() else None}
        all_runs.extend(read_runs(root, attention, lengths))
    if configs["softmax"] != configs["sparsemax"]:
        raise ValueError("The models have different training configurations")
    selected, unselected = [], []
    for attention in roots:
        for length in lengths:
            for width in configs[attention]["widths"]:
                candidates = [r for r in all_runs if r["attention"] == attention and
                              r["length"] == length and r["width"] == width]
                if len(candidates) != len(configs[attention]["learning_rates"]):
                    raise ValueError(f"Incomplete LR grid for {attention}, length {length}, width {width}")
                if not all(r["completed"] and r["epochs_observed"] == configs[attention]["epochs"]
                           for r in candidates):
                    raise ValueError(f"The validation sweep is not complete for {attention}, length {length}")
                if selection == "best":
                    choice = max(candidates, key=lambda r: (
                        r["best_validation"]["accuracy"], -r["best_validation"]["loss"]))
                else:
                    crossed = [r for r in candidates if r["milestones"]["99"] is not None]
                    eligible = crossed if selection == "first99" else [
                        r for r in crossed if r["milestones"]["99"]["sustained_to_end"]]
                    if not eligible:
                        unselected.append({"attention": attention, "length": length, "width": width,
                                           "target_percent": 99, "learning_rate": None,
                                           "selection_status": "99_percent_not_sustained" if crossed else
                                               "99_percent_unreached"})
                        continue
                    choice = min(eligible, key=lambda r: (
                        r["milestones"]["99"]["epoch"],
                        r["milestones"]["99"]["training_seconds"], r["learning_rate"]))
                selected.append(dict(choice, selection_status="selected"))
    return {"lengths": list(lengths), "thresholds_percent": list(THRESHOLDS),
            "config": configs["softmax"], "provenance": provenance,
            "selection_policy": selection,
            "selection": {
                "best": "Best validation accuracy, then validation loss; no test data used",
                "first99": "First observed 99% validation accuracy: earliest epoch, then measured "
                    "training seconds, then smaller LR. No test data used. Subsequent drops are allowed.",
                "stable99": "Earliest first crossing of 99% among runs with every subsequent validation "
                    "at or above 99% through the final epoch. Reject runs with any later drop, even "
                    "if they recover. Ties use measured training seconds, then smaller LR. No test data used."
            }[selection],
            "timing_scope": "Cumulative measured training loop time; includes lazy compilation, "
                "excludes validation, checkpoints, pauses, data setup, discarded partial epochs, "
                "and preceding LR candidates. Training plus validation sums the recorded timers.",
            "resolution": "First observed crossing after an epoch; no interpolation within epochs",
            "compilation_caveat": "Historical softmax includes eager and compiled runs; sparsemax "
                "is compiled. First-use compilation/cache warmup differs between LR candidates.",
            "selected_runs": selected, "unselected_groups": unselected, "all_runs": all_runs}


def render_csv(runs, unselected_groups=()):
    output = io.StringIO(newline="")
    fields = ("attention", "length", "width", "learning_rate", "selection_status", "threshold_percent", "epoch",
              "training_seconds", "training_plus_validation_seconds", "execution_modes",
              "sustained_to_end", "best_validation_accuracy", "last_validation_accuracy")
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for run in runs:
        for threshold in THRESHOLDS:
            hit = run["milestones"][str(threshold)]
            writer.writerow({"attention": run["attention"], "length": run["length"],
                "width": run["width"], "learning_rate": run["learning_rate"],
                "selection_status": run.get("selection_status", "candidate"),
                "threshold_percent": threshold, "epoch": None if hit is None else hit["epoch"],
                "training_seconds": None if hit is None else hit["training_seconds"],
                "training_plus_validation_seconds": None if hit is None else hit["training_plus_validation_seconds"],
                "execution_modes": "" if hit is None else "+".join(hit["execution_modes_until_crossing"]),
                "sustained_to_end": None if hit is None else hit["sustained_to_end"],
                "best_validation_accuracy": run["best_validation"]["accuracy"],
                "last_validation_accuracy": run["last_validation_accuracy"]})
    for group in unselected_groups:
        for threshold in THRESHOLDS:
            writer.writerow({"attention": group["attention"], "length": group["length"],
                             "width": group["width"], "learning_rate": None,
                             "selection_status": group["selection_status"], "threshold_percent": threshold})
    return output.getvalue()


def export_report(report, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    artifacts = {output.with_suffix(".json"): json.dumps(report, indent=2, allow_nan=False) + "\n",
                 output.with_suffix(".csv"): render_csv(report["selected_runs"], report["unselected_groups"]),
                 output.with_name(output.name + "_all_lr.csv"): render_csv(report["all_runs"])}
    for path, contents in artifacts.items():
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(contents)
        temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--softmax", type=Path, default=Path("runs/full"))
    parser.add_argument("--sparsemax", type=Path, default=Path("runs/sparsemax"))
    parser.add_argument("--lengths", nargs="+", type=int, default=[64, 128, 256])
    parser.add_argument("--selection", choices=("stable99", "first99", "best"), default="stable99",
                        help="Require 99 percent through the final epoch, allow drops, or select peak quality")
    parser.add_argument("--output", type=Path, default=Path("reports/validation_milestones"))
    args = parser.parse_args()
    report = build_report(args.softmax, args.sparsemax, args.lengths, args.selection)
    export_report(report, args.output)
    print("attention,length,learning_rate," + ",".join(f"seconds_to_{p}_percent" for p in THRESHOLDS))
    for run in report["selected_runs"]:
        values = ["unreached" if run["milestones"][str(p)] is None else
                  f"{run['milestones'][str(p)]['training_seconds']:.3f}" for p in THRESHOLDS]
        print(f"{run['attention']},{run['length']},{run['learning_rate']:.8g}," + ",".join(values))
    for group in report["unselected_groups"]:
        print(f"{group['attention']},{group['length']},none," +
              ",".join([group["selection_status"]] * len(THRESHOLDS)))


if __name__ == "__main__":
    main()
