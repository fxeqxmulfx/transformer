"""Export whole-history recovery frequencies without changing frozen success gates.

Convexifying Transformers, Section 4 supplies the modular task; the user
requested descriptive frequency metrics in addition to the persistence gate.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.synthetic_trainers.attention_layout import digest
from experiments.synthetic_trainers.persistence import PersistenceConfig
from experiments.synthetic_trainers.stability_recovery_series import timeline
from experiments.synthetic_trainers.stability_report import verify_archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Whole-history frequency figures require a fresh destination")
    summary = verify_archive(args.archive)
    plan = json.loads((args.archive / "plan.json").read_text())
    report = json.loads((args.archive / "measurements.json").read_text())
    width = plan["recovery_window_steps"]
    series = timeline(report, PersistenceConfig(**plan["criterion"]), window_steps=width)
    if series["metrics"] is None:
        raise ValueError("Whole post-onset frequency requires an observed long confirmation")
    metrics = series["metrics"]["joint"]
    windows = [w for w in metrics["fixed_grid_windows"]
               if w["complete_window"] and w["nominal_width_window"]]
    if not windows:
        raise ValueError("No complete nominal-width post-onset window")
    args.output.mkdir(parents=True)
    (args.output / "recovery-series.json").write_text(json.dumps(series, indent=2) + "\n")
    rows = []
    for score, data in series["metrics"].items():
        leading = data["leading_partial_width_window"]
        if leading is not None:
            rows.append({"score": score, "scope": "leading_partial_width", **leading})
        rows.extend({"score": score, "scope": "fixed_post_onset", **w}
                    for w in data["fixed_grid_windows"])
    with (args.output / "windows.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    figure, axes = plt.subplots(2, 1, figsize=(15, 8), constrained_layout=True)
    x = [(w["start"] + w["stop"]) / 2000 for w in windows]
    labels = [f'{w["start"] // 1000}–{w["stop"] // 1000}' for w in windows]
    fraction = [100 * w["failure_fraction"] for w in windows]
    onsets = [w["episode_onsets_per_10000_updates"] for w in windows]
    bars = axes[0].bar(x, fraction, width=width / 1250, color="#d55e00")
    for bar, window in zip(bars, windows):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + .4,
                     f'{window["failed_observations"]}/{window["observations"]}', ha="center", fontsize=8)
    axes[0].set(ylabel="Failed joint observations (%)", ylim=(0, max(fraction) + 3),
                title="All complete fixed post-onset windows; exact scheduled support")
    bars = axes[1].bar(x, onsets, width=width / 1250, color="#1678a5")
    for bar, window in zip(bars, windows):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + .05,
                     str(window["new_episode_onsets"]), ha="center", fontsize=8)
    axes[1].set(ylabel="New episodes per 10,000 updates", ylim=(0, max(onsets) + .5),
                xlabel="Fixed windows (thousands of updates); leading partial width retained in CSV",
                title="Adjacent missed canonical targets form one sampled episode")
    tail = summary["assessment"]["tail_start_step"] / 1000
    budget = series["frozen_budget"] / 1000
    for axis in axes:
        axis.axvspan(tail, budget, color="#999999", alpha=.12, label="frozen final window")
        axis.set_xticks(x, labels, rotation=45, ha="right", fontsize=8)
        axis.set_xlim(windows[0]["start"] / 1000, budget)
        axis.grid(axis="y", alpha=.2)
        axis.set_axisbelow(True)
        axis.legend(loc="upper right")
    config = report["plan"]["config"]
    figure.suptitle(f'Native AdamW · mod {config["prime"]} · fraction {config["train_fraction"]:g} · '
                   f'lr {config["learning_rate"]:g} · seeds {config["seed"]}/{config["data_seed"]}\n'
                   f'Complete {config["steps"]:,} updates; descriptive frequencies, no permanent-stability claim')
    for extension in ("png", "pdf"):
        figure.savefig(args.output / f"series.{extension}", dpi=150)
    plt.close(figure)
    provenance = {"archive": str(args.archive), "measurement_sha256": digest(args.archive / "measurements.json"),
                  "archive_artifact_manifest_sha256": digest(args.archive / "artifact-hashes.json"),
                  "series_source_sha256": digest(Path("experiments/synthetic_trainers/stability_recovery_series.py")),
                  "plot_source_sha256": digest(Path(__file__)), "window_steps": width,
                  "scope": series["scope"], "leading_partial_window_retained": True}
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    hashes = {p.name: digest(p) for p in args.output.iterdir()}
    (args.output / "artifact-hashes.json").write_text(json.dumps({"files": hashes}, indent=2) + "\n")
    print(json.dumps({"complete_nominal_windows": len(windows), "leading_partial_window_retained": True}))


if __name__ == "__main__":
    main()
