"""Standalone figures for saved baseline summaries (Matplotlib, no Torch)."""

import argparse
import json
from pathlib import Path
import statistics

from .plots import pyplot


def save(figure, output, name, plt):
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(output / f"{name}.{suffix}", dpi=160)
    plt.close(figure)


def render(directory, output):
    directory, output = Path(directory), Path(output)
    source = directory / "summary.json"
    if not source.exists():
        source = directory / "measurements.json"
    summary = json.loads(source.read_text())
    output.mkdir(parents=True, exist_ok=True)
    plt = pyplot()
    suite = [row for row in summary["runs"] if row["phase"] == "suite"]
    if suite:
        values = []
        for row in suite:
            probes = [name for name in row["test_final"] if name.startswith("length-")]
            probes.sort(key=lambda name: int(name.split("-")[-1]))
            values.append([row["train_sequence_accuracy"], row["novel_validation_accuracy"] if row["novel_validation_accuracy"] is not None else float("nan"),
                           row["test_id_accuracy"], *[row["test_final"][name]["sequence_accuracy"] for name in probes]])
        figure, axis = plt.subplots(figsize=(9, 13))
        chart = axis.imshow(values, cmap="viridis", vmin=0, vmax=1, aspect="auto")
        axis.set_yticks(range(len(suite)), [row["variant"] for row in suite], fontsize=8)
        axis.set_xticks(range(5), ["Train\n(teacher forced)", "Novel ID\nvalidation", "ID test", "2x length\ntest", "4x length\ntest"], fontsize=9)
        for i, line in enumerate(values):
            for j, value in enumerate(line):
                label = f"{value * 100:.0f}" if value == value else "n/a"
                axis.text(j, i, label, ha="center", va="center", fontsize=8, color="black" if value > .6 else "white")
        figure.colorbar(chart, ax=axis, label="Complete-example accuracy")
        axis.set_title(f"AMSGradW + softmax GPTMini: {len(suite)} variants, {suite[0]['steps']:,} updates, seed {suite[0]['seed']}\n"
                       "Final checkpoint; hard-carry/position probes are in the JSON/CSV reports", fontsize=10)
        save(figure, output, "suite-accuracy", plt)
    transition_rows = [row for row in summary["runs"] if row["phase"] == "transitions"]
    variants = list(dict.fromkeys(row["variant"] for row in transition_rows))
    if variants:
        figure, axes = plt.subplots(2, len(variants), figsize=(5 * len(variants), 8), squeeze=False)
        for column, variant in enumerate(variants):
            rows = [row for row in transition_rows if row["variant"] == variant]
            histories = [row["history"] if "history" in row else
                         [json.loads(line) for line in (directory / row["source_result"]).with_name("history.jsonl").read_text().splitlines()]
                         for row in rows]
            steps = [row["step"] for row in histories[0]]
            keys = [("train", "Train (teacher forced)"), ("validation_novel", "Novel ID validation"),
                    *((name, name) for name in histories[0][0]["validation_ood"])]
            for key, label in keys:
                for axis, metric in zip(axes[:, column], ("example_loss", "sequence_accuracy")):
                    curves = [[(point[key] if key in point else point["validation_ood"][key])[metric] for point in history]
                              for history in histories]
                    means = [statistics.mean(values) for values in zip(*curves)]
                    deviations = [statistics.stdev(values) if len(values) > 1 else 0 for values in zip(*curves)]
                    line, = axis.plot(steps, means, label=label, linestyle="--" if key.startswith("length-") else "-")
                    axis.fill_between(steps, [a - b for a, b in zip(means, deviations)],
                                      [a + b for a, b in zip(means, deviations)], color=line.get_color(), alpha=.12)
            axes[0, column].set_title(f"{variant}: {len(rows)} initialization seeds")
            axes[0, column].set_ylabel("Mean per-example CE (nats / target)")
            axes[1, column].set_ylabel("Complete-example accuracy")
            axes[1, column].set_ylim(-.02, 1.02)
            for axis in axes[:, column]:
                axis.set_xlabel("Optimizer updates")
                axis.grid(alpha=.2)
                axis.legend(fontsize=8)
        figure.suptitle("AMSGradW + softmax GPTMini: delayed generalization probes\nMean +/- sample SD; disjoint ID pools, one fixed data seed", fontsize=11)
        save(figure, output, "transition-curves", plt)
    capacity = [row for row in summary["aggregates"] if row["phase"] == "capacity"]
    capacity.sort(key=lambda row: row["parameters"]["mean"])
    if capacity:
        figure, axes = plt.subplots(1, 3, figsize=(14, 4.5))
        x = [row["parameters"]["mean"] for row in capacity]
        for axis, key, label in zip(axes, ("test_id_loss", "test_id_accuracy", "train_sequence_accuracy"),
                                   ("Clean test CE (nats / target)", "Clean test complete-example accuracy", "Observed train complete-example accuracy")):
            y = [row[key]["mean"] for row in capacity]
            std = [row[key]["std"] for row in capacity]
            axis.errorbar(x, y, yerr=std, marker="o", capsize=4)
            axis.set_xscale("log")
            axis.set_xticks(x, [f"{value / 1000:.1f}k" for value in x])
            axis.minorticks_off()
            axis.set_xlabel("Parameters (log scale)\nTied weights counted once")
            axis.set_ylabel(label)
            axis.grid(alpha=.2)
        axes[1].set_ylim(-.02, 1.02)
        axes[2].set_ylim(-.02, 1.02)
        capacity_runs = [row for row in summary["runs"] if row["phase"] == "capacity"]
        realized = capacity_runs[0]["noise_targets_changed"] / capacity_runs[0]["train_examples"]
        figure.suptitle(f"Parity, requested noise {capacity_runs[0]['noise']:.0%} (realized {realized:.0%}), {capacity_runs[0]['steps']:,} updates\n"
                       f"Mean +/- sample SD across {capacity[0]['runs']} initialization seeds; one fixed data seed", fontsize=11)
        save(figure, output, "capacity-curves", plt)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    print(render(args.directory, args.output or args.directory / "plots"))


if __name__ == "__main__":
    main()
