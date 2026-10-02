"""Export verified complete histories and descriptive recovery windows.

Convexifying Transformers, Section 4 supplies the modular task. Window and
episode statistics are user-requested follow-ups, with unchanged success gates.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.synthetic_trainers.persistence import PersistenceConfig
from experiments.synthetic_trainers.stability_recovery import describe
from experiments.synthetic_trainers.stability_report import verify_archive


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--window-steps", type=int, default=10000)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Recovery figures require a fresh destination")
    verify_archive(args.archive)
    plan = json.loads((args.archive / "plan.json").read_text())
    report = json.loads((args.archive / "measurements.json").read_text())
    result = describe(report, PersistenceConfig(**plan["criterion"]), window_steps=args.window_steps)
    config, points = report["plan"]["config"], report["history"]
    joint = result["metrics"]["joint"]
    windows = joint["fixed_tail_windows"]
    args.output.mkdir(parents=True)
    (args.output / "recovery-metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    with (args.output / "windows.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(windows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(windows)
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    x = [p["step"] / 1000 for p in points]
    for split, color in (("train", "#1678a5"), ("heldout", "#d55e00")):
        axes[0, 0].plot(x, [p[split]["accuracy"] for p in points], color=color, label=split)
    axes[0, 0].axhline(result["criterion"]["target"], color="black", ls=":", label="99% target")
    streak = joint["final_target_streak"]
    if streak["start"] is not None:
        axes[0, 0].axvspan(streak["start"] / 1000, streak["end"] / 1000,
                           color="#009e73", alpha=.12, label="final joint-target streak")
    axes[0, 0].set(xlabel="Updates (thousands)", ylabel="Complete RHS accuracy", ylim=(-.02, 1.03),
                   title=f"All {len(points):,} canonical observations; no smoothing")
    axes[0, 0].legend(loc="lower right")
    labels = [f'{w["start"] // 1000}–{w["stop"] // 1000}' for w in windows]
    fractions = [100 * w["failure_fraction"] for w in windows]
    bars = axes[0, 1].bar(labels, fractions, color="#d55e00")
    for bar, window in zip(bars, windows):
        axes[0, 1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + .4,
                        f'{window["failed_observations"]}/{window["observations"]}', ha="center")
    axes[0, 1].set(xlabel="Fixed tail windows (thousands of updates)", ylabel="Failed joint observations (%)",
                   ylim=(0, max(fractions, default=0) + 4), title="Missed targets / exact scheduled support")
    onsets = [w["episode_onsets_per_10000_updates"] for w in windows]
    if all(value is not None for value in onsets):
        bars = axes[1, 0].bar(labels, onsets, color="#1678a5")
        for bar, value in zip(bars, onsets):
            axes[1, 0].text(bar.get_x() + bar.get_width() / 2, value + .06, f"{value:g}", ha="center")
        axes[1, 0].set_ylim(0, max(onsets, default=0) + .5)
    else:
        axes[1, 0].text(.5, .5, "No long confirmation: episode rate unavailable", ha="center", transform=axes[1, 0].transAxes)
    axes[1, 0].set(xlabel="Fixed tail windows (thousands of updates)", ylabel="New episodes per 10,000 updates",
                   title="Consecutive missed targets form one episode")
    episodes = joint["episodes_after_long_onset"]
    episode_labels = [f'{e["first_failure"]:,}' for e in episodes]
    durations = [e["sampled_steps_until_first_recovery"] for e in episodes]
    if episodes and all(value is not None for value in durations):
        bars = axes[1, 1].bar(episode_labels, durations, color="#009e73")
        for bar, episode in zip(bars, episodes):
            axes[1, 1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                            f'min {100 * episode["minimum_accuracy"]:.2f}%', ha="center")
        axes[1, 1].set_ylim(0, max(durations) * 1.2 + 50)
    axes[1, 1].set(xlabel="First failed canonical update", ylabel="Sampled steps to first recovery",
                   title="Recovery intervals and minimum joint accuracy")
    for axis in axes.flat:
        axis.grid(axis="y", alpha=.2)
        axis.set_axisbelow(True)
    figure.suptitle(f'Native AdamW · mod {config["prime"]} · train fraction {config["train_fraction"]:g} · '
                    f'lr {config["learning_rate"]:g} · seeds {config["seed"]}/{config["data_seed"]}\n'
                    f'Complete {config["steps"]:,}-update calibration; strict persistence: '
                    f'{result["frozen_persistent_final_performance"]}; descriptive sampled recovery', fontsize=13)
    for extension in ("png", "pdf"):
        figure.savefig(args.output / f"recovery.{extension}", dpi=150)
    plt.close(figure)
    provenance = {"archive": str(args.archive), "measurement_sha256": digest(args.archive / "measurements.json"),
        "archive_artifact_manifest_sha256": digest(args.archive / "artifact-hashes.json"),
        "metric_source_sha256": digest(Path("experiments/synthetic_trainers/stability_recovery.py")),
        "plot_source_sha256": digest(Path(__file__)), "window_steps": args.window_steps,
        "canonical_observations": len(points), "scope": "complete_unsmoothed_history; sampled_descriptive_metrics; no_future_stability_claim"}
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    hashes = {p.name: digest(p) for p in args.output.iterdir()}
    (args.output / "artifact-hashes.json").write_text(json.dumps({"files": hashes}, indent=2) + "\n")
    print(json.dumps(provenance))


if __name__ == "__main__":
    main()
