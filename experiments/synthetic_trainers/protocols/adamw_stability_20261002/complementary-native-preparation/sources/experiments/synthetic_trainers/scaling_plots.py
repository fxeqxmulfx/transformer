"""Export standalone scaling figures from saved measurements, without PyTorch."""

import argparse
import json
from pathlib import Path


def export(figure, directory, name):
    directory.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        figure.savefig(directory / f"{name}.{extension}", dpi=170, bbox_inches="tight")


def errorbar(axis, rows, field, label, **style):
    axis.errorbar([row["width"] for row in rows], [row[field]["mean"] for row in rows],
                  yerr=[row[field]["std"] for row in rows], label=label, capsize=4, marker="o", **style)


def calibration_figure(summary, output, plt):
    rows = sorted((row for row in summary["aggregates"] if row["phase"] == "dd-calibration"), key=lambda row: row["train_examples"])
    if not rows:
        return
    figure, axes = plt.subplots(1, 2, figsize=(10.6, 3.8), constrained_layout=True)
    sizes = [row["train_examples"] for row in rows]
    axes[0].plot(sizes, [row["train_accuracy"]["mean"] for row in rows], "o-", label="Observed noisy train")
    axes[0].plot(sizes, [row["train_clean_accuracy"]["mean"] for row in rows], "s--", label="Clean labels on train inputs")
    axes[0].plot(sizes, [row["id_test_accuracy"]["mean"] for row in rows], "^-", label="Clean ID test")
    axes[0].axhline(.99, linestyle=":", color="grey", label="Fit threshold (strictly >99%)")
    axes[0].set(ylabel="Complete-example accuracy", ylim=(0, 1.06))
    axes[0].legend(fontsize=8)
    axes[1].plot(sizes, [row["id_test_loss"]["mean"] for row in rows], "o-", color="#d55e00")
    axes[1].set(ylabel="Clean ID test CE (nats / target)")
    for axis in axes:
        axis.set_xscale("log", base=4)
        axis.set_xticks(sizes, [str(size) for size in sizes])
        axis.set_xlabel("Training examples N (nested pools)")
        axis.grid(alpha=.2)
    choice = summary["plan"]["sample_choice"]
    steps = summary["plan"]["dd_training"]["steps"]
    figure.suptitle(f"16-bit noisy parity: width 64, depth 2, {steps:,} updates; next N={choice['chosen_sample_size']} from train fit", fontsize=11)
    export(figure, output, "fit-calibration")
    plt.close(figure)


def widths_figure(summary, output, plt):
    rows = sorted((row for row in summary["aggregates"] if row["phase"] == "dd-widths"), key=lambda row: row["width"])
    if not rows:
        return
    figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.1), constrained_layout=True)
    errorbar(axes[0], rows, "id_test_loss", "Final clean test", color="#d55e00")
    axes[0].set_ylabel("Clean ID test CE (nats / target)")
    axes[0].legend(fontsize=8)
    errorbar(axes[1], rows, "train_accuracy", "Observed noisy train", color="#0072b2")
    errorbar(axes[1], rows, "id_test_accuracy", "Clean ID test", color="#d55e00")
    axes[1].axhline(.5, color="grey", linestyle=":", label="Parity chance accuracy")
    axes[1].set(ylabel="Complete-example accuracy", ylim=(.35, 1.07))
    axes[1].legend(fontsize=8)
    for axis in axes:
        axis.set_xscale("log", base=2)
        axis.set_xticks([row["width"] for row in rows], [str(row["width"]) for row in rows])
        axis.set_xlabel("Model width (depth 2, eight heads)")
        axis.grid(alpha=.2)
    fitted = ", ".join(f"{row['width']}: {row['fitted_runs']}/{row['runs']}" for row in rows)
    steps = summary["plan"]["dd_training"]["steps"]
    seeds = len(summary["plan"]["width_grid"]["seeds"])
    figure.suptitle(f"N={rows[0]['train_examples']}, {steps:,} updates; means ± sample SD across {seeds} initializations\nFitted runs by width (>99% train accuracy): {fitted}", fontsize=10)
    export(figure, output, "parity-widths")
    plt.close(figure)


def copy_figure(summary, output, plt):
    rows = sorted((row for row in summary["runs"] if row["phase"] == "copy-scaling"), key=lambda row: (row["width"], row["layers"]))
    if not rows:
        return
    figure, axes = plt.subplots(2, 2, figsize=(12.2, 8.0), constrained_layout=True)
    colors = ("#0072b2", "#009e73", "#e69f00", "#d55e00")
    names = [f"w{row['width']} / L{row['layers']}\n{row['parameters']/1e6:.3f}M parameters" for row in rows]
    groups = ("ID test", "Novel ID\nvalidation", "OOD 64", "OOD 128")
    for index, (row, color) in enumerate(zip(rows, colors)):
        values = [row["test_final"]["in_distribution"]["sequence_accuracy"], row["novel_validation_accuracy"],
                  row["test_final"]["length-64"]["sequence_accuracy"], row["test_final"]["length-128"]["sequence_accuracy"]]
        axes[0, 0].bar([x + (index - 1.5) * .18 for x in range(4)], values, width=.17, color=color, label=f"w{row['width']}/L{row['layers']}")
        history = row["history"]
        axes[0, 1].plot([point["step"] for point in history], [point["train"]["sequence_accuracy"] for point in history], color=color, marker="o")
        axes[0, 1].plot([point["step"] for point in history], [point["validation_novel"]["sequence_accuracy"] for point in history], color=color, linestyle="--", marker="s", label=f"w{row['width']}/L{row['layers']}")
        token_values = [row["test_final"][probe]["token_accuracy"] for probe in ("in_distribution", "length-64", "length-128")]
        axes[1, 0].bar([x + (index - 1.5) * .18 for x in range(3)], token_values, width=.17, color=color)
    axes[0, 0].set_xticks(range(4), groups)
    axes[0, 0].set(ylabel="Free-rollout complete-example accuracy", ylim=(0, 1.04))
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].set(xlabel="Updates", ylabel="Complete-example accuracy", ylim=(0, 1.04), title="Solid: teacher-forced train; dashed: novel ID rollout")
    axes[0, 1].legend(fontsize=8)
    axes[1, 0].set_xticks(range(3), ("ID test", "OOD length 64", "OOD length 128"))
    axes[1, 0].set(ylabel="Free-rollout token accuracy", ylim=(0, 1.04))
    axes[1, 1].bar(range(len(rows)), [row["training_seconds"] for row in rows], color=colors[:len(rows)])
    axes[1, 1].set_xticks(range(len(rows)), names, fontsize=8)
    axes[1, 1].set_ylabel("GPU training-update seconds")
    for row_index, row in enumerate(rows):
        axes[1, 1].annotate(f"{row['training_seconds']:.0f}s", (row_index, row["training_seconds"]), ha="center", va="bottom", fontsize=9)
    for axis in axes.flat:
        axis.grid(axis="y", alpha=.2)
    coverage = summary["plan"]["actual_copy_motif_coverage"]
    steps = summary["plan"]["copy_training"]["steps"]
    figure.suptitle(f"Binary copy, lengths 1–32, N={rows[0]['train_examples']:,}, {steps:,} updates, AMSGradW + softmax\nEight heads throughout; one paired model seed; actual motif coverage {coverage['observed_events']}/{coverage['events']}", fontsize=11)
    export(figure, output, "copy-scaling")
    plt.close(figure)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    summary = json.loads((args.directory / "measurements.json").read_text())
    output = args.directory / "plots"
    calibration_figure(summary, output, plt)
    widths_figure(summary, output, plt)
    copy_figure(summary, output, plt)
    print(output)


if __name__ == "__main__":
    main()
