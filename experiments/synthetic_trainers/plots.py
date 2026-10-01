"""Export research plots from saved reports; requires Matplotlib, not PyTorch."""

import argparse
import json
from pathlib import Path

from .specs import TaskSpec


def pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def render_run(directory, output):
    plt = pyplot()
    directory, output = Path(directory), Path(output)
    report = json.loads((directory / "result.json").read_text())
    history = [json.loads(line) for line in (directory / "history.jsonl").read_text().splitlines()]
    if "study" not in report:
        raise ValueError("Run plots require a study profile")
    random_control = report["task"] == "random_lm"
    figure, axes = plt.subplots(1, 3 if random_control else 2, figsize=(15 if random_control else 11, 4.5))
    epochs = [row["epochs_seen"] for row in history]
    metric = report["target_metric"]
    validation_label = "validation clean (free rollout)" if TaskSpec(**report["provenance"]["task"]).generative else "validation clean"
    for key, label in (("train", "train observed (teacher forced)"), ("train_clean", "train clean (teacher forced)"),
                       ("validation", validation_label)):
        axes[0].plot(epochs, [row[key]["example_loss"] for row in history], label=label)
        accuracy = "sequence_accuracy" if metric == "final_answer_accuracy" and key.startswith("train") else metric
        axes[1].plot(epochs, [row[key][accuracy] for row in history], label=label)
    for name in history[0]["validation_ood"]:
        axes[0].plot(epochs, [row["validation_ood"][name]["example_loss"] for row in history], linestyle="--", label=name)
        axes[1].plot(epochs, [row["validation_ood"][name][metric] for row in history], linestyle="--", label=name)
    transition = report["study"]["generalization_transition"]
    if transition.get("train_validation_overlap_inputs", 0) and history[0].get("validation_novel"):
        axes[1].plot(epochs, [row["validation_novel"][metric] for row in history], linestyle=":", label="validation novel inputs")
    for name, overlap in transition.get("probe_input_overlap", {}).items():
        if overlap and history[0].get("validation_ood_novel", {}).get(name):
            axes[1].plot(epochs, [row["validation_ood_novel"][name][metric] for row in history], linestyle=":", label=f"{name} novel inputs")
    for key, label in (("observed_train_fit_step", "train fit"), ("generalization_step", "confirmed transfer onset")):
        step = transition[key]
        if step is not None:
            epoch = next(row["epochs_seen"] for row in history if row["step"] == step)
            for axis in axes[:2]:
                axis.axvline(epoch, color="gray", linestyle=":" if key.endswith("fit_step") else "-.", label=label)
    axes[0].set_ylabel("Mean per-example cross entropy (nats / target)")
    axes[1].set_ylabel(metric.replace("_", " ").capitalize())
    axes[1].set_ylim(-0.02, 1.02)
    if random_control:
        axes[2].plot(epochs, [row["compression"]["net_gain_bits"] for row in history], label="net sequence coding gain")
        axes[2].plot(epochs, [row["compression"]["mixture_gain_bits"] for row in history], label="normalized mixture gain")
        axes[2].axhline(0, color="gray", linewidth=0.8)
        axes[2].set_ylabel("Measured coding gain (bits)")
    for axis in axes:
        axis.set_xlabel("Examples seen / finite training pool size")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=7)
    figure.suptitle(f"{report['variant']}: {report['parameters']:,} parameters, noise {report['study']['noise']['rate']:g}")
    figure.tight_layout()
    output.mkdir(parents=True, exist_ok=True)
    path = output / "learning-curves.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return [path]


def render_sweep(directory, output):
    plt = pyplot()
    directory, output = Path(directory), Path(output)
    report = json.loads((directory / "sweep.json").read_text())
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    settings = sorted({(row["layers"], row["label_noise"]) for row in report["aggregates"]})
    for layers, noise in settings:
        rows = [row for row in report["aggregates"] if row["layers"] == layers and row["label_noise"] == noise]
        figure, axes = plt.subplots(2, 2, figsize=(11, 8))
        for size in sorted({row["train_examples"] for row in rows}):
            curve = sorted((row for row in rows if row["train_examples"] == size), key=lambda row: row["width"])
            x = [row["parameters"]["mean"] for row in curve]
            y = [row["final_loss"]["mean"] for row in curve]
            std = [row["final_loss"]["std"] for row in curve]
            line, = axes[0, 0].plot(x, y, marker="o", label=f"N={size}, final")
            axes[0, 0].fill_between(x, [a - b for a, b in zip(y, std)], [a + b for a, b in zip(y, std)], color=line.get_color(), alpha=0.15)
            axes[0, 0].plot(x, [row["selected_loss"]["mean"] for row in curve], color=line.get_color(), linestyle=":", label=f"N={size}, selected")
        has_capacity = any(row["net_gain_bits"] is not None for row in rows)
        for width in sorted({row["width"] for row in rows}):
            curve = sorted((row for row in rows if row["width"] == width), key=lambda row: row["train_examples"])
            x = [row["train_examples"] for row in curve]
            axes[0, 1].plot(x, [row["final_loss"]["mean"] for row in curve], marker="o", label=f"width={width}")
            axes[1, 0].plot(x, [row["fitted_runs"] / row["runs"] for row in curve], marker="o", label=f"width={width}")
            if has_capacity:
                axes[1, 1].plot(x, [row["net_gain_bits"]["mean"] for row in curve], marker="o", label=f"width={width}")
            else:
                axes[1, 1].plot(x, [row["final_error"]["mean"] for row in curve], marker="o", label=f"width={width}")
        axes[0, 0].set_xlabel("Model parameters")
        axes[0, 0].set_ylabel("Clean test loss: mean +/- sample SD")
        axes[0, 1].set_ylabel("Clean test loss, final checkpoint")
        axes[1, 0].set_ylabel("Fraction of runs fitting observed train labels")
        axes[1, 0].set_ylim(-0.02, 1.02)
        axes[1, 1].set_ylabel("Net coding gain (bits), final" if has_capacity else "Clean test task error, final")
        for axis in (axes[0, 1], axes[1, 0], axes[1, 1]):
            axis.set_xlabel("Finite training pool size")
        for axis in axes.flat:
            axis.grid(alpha=0.2)
            axis.legend(fontsize=7)
        figure.suptitle(f"Layers {layers}, noise {noise:g}; final and validation-selected checkpoints")
        figure.tight_layout()
        path = output / f"sweep-layers-{layers}-noise-{noise}.png"
        figure.savefig(path, dpi=160)
        plt.close(figure)
        paths.append(path)
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report_directory", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    output = args.output or args.report_directory / "plots"
    render = render_sweep if (args.report_directory / "sweep.json").is_file() else render_run
    for path in render(args.report_directory, output):
        print(path)


if __name__ == "__main__":
    main()
