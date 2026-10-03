"""Export paper reproduction curves as standalone PNG/PDF without PyTorch."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics

from .paper_phases import diagnose


def export(figure, directory, name):
    directory.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        figure.savefig(directory / f"{name}.{extension}", dpi=180, bbox_inches="tight")


def rff_figures(report, directory, plt):
    figure, axes = plt.subplots(2, 2, figsize=(11.6, 7.2), constrained_layout=True)
    for column, curve in enumerate(report["curves"]):
        axis = curve["axis"]
        fixed_key, fixed_value = next(iter(curve["fixed"].items()))
        points = sorted(curve["points"], key=lambda row: row[axis])
        xs = [row[axis] for row in points]
        color = ("#0072b2", "#d55e00")[column]
        for seed in report["plan"]["seeds"]:
            rows = sorted((row for row in report["runs"] if row["seed"] == seed and row[fixed_key] == fixed_value), key=lambda row: row[axis])
            axes[0, column].plot([row[axis] for row in rows], [row["test"]["error"] for row in rows], color=color, alpha=.23, linewidth=1)
            axes[1, column].plot([row[axis] for row in rows], [row["test"]["MSE"] for row in rows], color=color, alpha=.23, linewidth=1)
        for row, metric in ((0, "error"), (1, "MSE")):
            axes[row, column].errorbar(xs, [point["test"][metric]["mean"] for point in points],
                yerr=[point["test"][metric]["std"] for point in points], color=color,
                marker="o", markersize=3, capsize=3, label="Test mean ± sample SD")
            axes[row, column].axvline(fixed_value, color="grey", linestyle=":", label="n=d interpolation boundary")
            axes[row, column].grid(alpha=.2)
        axes[0, column].plot(xs, [point["train"]["error"]["mean"] for point in points],
                             color="#555555", linestyle="--", label="Train classification error")
        axes[0, column].set(ylim=(-.015, 1), ylabel="Classification error", title=f"{axis.capitalize()}-wise: {fixed_key}={fixed_value:,}")
        axes[0, column].legend(fontsize=8)
        axes[1, column].set(yscale="log", ylabel="Test full complex MSE", xlabel="Training examples n" if axis == "samples" else "Random feature width d")
    seeds = len(report["plan"]["seeds"])
    figure.suptitle(f"Fashion-MNIST, exp(-i*x) features, zero-initialized gradient-flow limit\n{seeds} paired data/feature seeds; all 10,000 official test images; normalization uint8/255", fontsize=12)
    export(figure, directory, "random-feature-slices")
    plt.close(figure)


def modular_figures(report, directory, plt):
    grouped = defaultdict(list)
    for row in report.get("runs", [report]):
        config = row["plan"]["config"]
        grouped[config["model"], config["optimizer"]].append(row)
    figure, axes = plt.subplots(3, len(grouped), squeeze=False, figsize=(5 * len(grouped), 10.2), constrained_layout=True)
    panels = ((0, "accuracy", False), (1, "accuracy", True), (2, "loss", False))
    order = sorted(grouped, key=lambda pair: (pair[0] != "reference", pair[1] != "adamw"))
    for column, key in enumerate(order):
        runs = grouped[key]
        histories = [[point for point in row["history"] if point["step"] > 0] for row in runs]
        common = sorted(set.intersection(*({point["step"] for point in history} for history in histories)))
        lookup = [{point["step"]: point for point in history} for history in histories]
        for split, color, style in (("train", "#0072b2", "--"), ("heldout", "#d55e00", "-")):
            for history in histories:
                for row, metric, error in panels:
                    values = [1 - point[split][metric] if error else point[split][metric] for point in history]
                    axes[row, column].plot([point["step"] for point in history], values,
                                           color=color, linestyle=style, alpha=.2, linewidth=.7)
            for row, metric, error in panels:
                values = [[1 - table[step][split][metric] if error else table[step][split][metric]
                           for table in lookup] for step in common]
                means = [statistics.mean(points) for points in values]
                deviations = [statistics.stdev(points) if len(points) > 1 else 0 for points in values]
                axes[row, column].plot(common, means, color=color, linestyle=style, label=split.capitalize())
                axes[row, column].fill_between(common, [mean - sd for mean, sd in zip(means, deviations)],
                                               [mean + sd for mean, sd in zip(means, deviations)], color=color, alpha=.1)
        for run in runs:
            phase = diagnose(run)
            for field, color in (("sustained_train_fit", "#0072b2"), ("sustained_heldout_target", "#d55e00")):
                event = phase[field]
                if event and event["onset"]:
                    for row in (0, 1):
                        axes[row, column].axvline(event["onset"], color=color, linestyle=":", alpha=.4, linewidth=.7)
        axes[0, column].axhline(.99, color="grey", linestyle=":", linewidth=.8)
        seed_label = "seed" if len(runs) == 1 else "seeds"
        axes[0, column].set(ylim=(-.02, 1.025), ylabel="Complete RHS accuracy", title=f"{key[0]} / {key[1]}\n{len(runs)} initialization {seed_label}")
        axes[0, column].legend(fontsize=8)
        axes[1, column].set(ylim=(-.02, 1.025), ylabel="Complete RHS error", xlabel="Updates")
        axes[1, column].axhline(.01, color="grey", linestyle=":", linewidth=.8)
        axes[2, column].set(ylabel="Cross entropy per RHS token (nats)", xlabel="Updates")
        for row in (0, 1, 2):
            axes[row, column].set_xscale("log")
            axes[row, column].grid(alpha=.2)
    first = next(iter(grouped.values()))[0]["plan"]
    figure.suptitle(f"Mod-{first['config']['prime']} division, {first['corpus']['train_examples']:,} train / {first['corpus']['heldout_examples']:,} unseen equations\nFixed budgets; means with sample SD; dotted event markers use 99% / two observations", fontsize=12)
    export(figure, directory, "modular-generalization")
    plt.close(figure)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("rff", "modular"))
    parser.add_argument("directory", type=Path)
    args = parser.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    report = json.loads((args.directory / "measurements.json").read_text())
    output = args.directory / "plots"
    (rff_figures if args.kind == "rff" else modular_figures)(report, output, plt)
    print(output)


if __name__ == "__main__":
    main()
