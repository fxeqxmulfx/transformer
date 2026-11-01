"""Export the complete pair-task learning curves as standalone figures.

Uses only the committed audit JSON, including early and continuation points.
The initial screen endpoint is marked; neither seeds nor checkpoints are
selected by accuracy. Matplotlib is needed only to regenerate this figure.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent


def main():
    report = json.loads((HERE / "results.json").read_text())
    cases = {row["label"]: row for row in report["cases"]}
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
    for row, prime in enumerate((97, 193)):
        for column, shape in enumerate(("w256d1", "w128d4")):
            axis = axes[row, column]
            for task, color in (("division", "#1368a8"), ("mixed", "#b44d16")):
                case = cases[f"pairs-{task}-p{prime}-{shape}"]
                curve = [p for p in case["learning_curve"] if p["step"] > 0]
                steps = [p["step"] for p in curve]
                axis.plot(steps, [100*p["heldout"]["accuracy"] for p in curve],
                          color=color, label=f"{task}: heldout", linewidth=1.8)
                axis.plot(steps, [100*p["train"]["accuracy"] for p in curve],
                          color=color, linestyle=":", alpha=.6, linewidth=1,
                          label=f"{task}: train")
            axis.axvline(2000, color="#555555", linewidth=.8, linestyle="--")
            axis.axhline(99, color="#999999", linewidth=.6, linestyle=":")
            axis.set_xscale("log")
            axis.set_title(f"P={prime}, width {'256' if column==0 else '128'}, depth {1 if column==0 else 4}")
            axis.set_ylim(-2, 103)
            axis.grid(alpha=.18)
    axes[0, 0].legend(fontsize=8, loc="center left")
    for axis in axes[:, 0]:
        axis.set_ylabel("Accuracy (%)")
    for axis in axes[-1]:
        axis.set_xlabel("Absolute optimizer updates (log scale)")
    fig.suptitle("Original normalized RoPE transformer: full-state Schedule-Free continuations")
    fig.text(.5, .015, "Dashed line: 2,000-update screen. All paired endpoints retained; approximate parameter matching, unequal compute.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0,.04,1,.95))
    fig.savefig(HERE / "learning_curves.png", dpi=180)
    fig.savefig(HERE / "learning_curves.svg")


if __name__ == "__main__":
    main()
