"""Complete fixed-budget calibration comparisons, one panel per mechanism."""

import json

from .stability_plots import export


def render_comparison(directory, summary, *, title=None, filename="calibration-comparison"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = summary["rows"]
    figure, axes = plt.subplots(3, len(rows), figsize=(4 * len(rows), 9), squeeze=False, constrained_layout=True)
    for column, row in enumerate(rows):
        report = json.loads((directory / "runs" / row["name"] / "measurements.json").read_text())
        history = report["history"]
        xs = [point["step"] for point in history]
        for split, color in (("train", "#0072b2"), ("heldout", "#d55e00")):
            axes[0, column].plot(xs, [point[split]["accuracy"] for point in history], label=split, color=color)
        axes[0, column].axhline(report["plan"]["config"]["target"], color="grey", linestyle=":", linewidth=.7)
        axes[0, column].set(ylim=(-.02, 1.025), ylabel="Complete RHS accuracy", title=row["name"])
        axes[0, column].legend(fontsize=8)
        for metric, color in (("answer_loss", "#d55e00"), ("EOS_loss", "#009e73")):
            axes[1, column].plot(xs, [max(1e-12, p["heldout"][metric]) for p in history], label=metric, color=color)
        axes[1, column].set(yscale="log", ylabel="Held-out CE (display floor 1e-12)")
        axes[1, column].legend(fontsize=8)
        axes[2, column].plot([p["training_seconds"] for p in history], [p["heldout"]["accuracy"] for p in history], color="#d55e00")
        axes[2, column].set(ylim=(-.02, 1.025), xlabel="Measured training seconds", ylabel="Held-out complete RHS accuracy")
        for axis in axes[:2, column]:
            axis.axvspan(max(0, row["steps"] - summary["criterion"]["tail_steps"]), row["steps"], color="grey", alpha=.08)
            axis.set_xlabel("Updates")
        for axis in axes[:, column]:
            axis.grid(alpha=.2)
    figure.suptitle(title or f"{summary.get('optimizer_label', 'Raw AMSGradW')} / unchanged GPTMini: paired calibration controls\n"
                   "One initialization and one common split per recipe; all complete budgets and failures", fontsize=12)
    export(figure, directory / "plots", filename)
    plt.close(figure)
