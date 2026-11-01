"""Export all retained learning curves without choosing seeds or checkpoints.

Matplotlib is an artifact-rendering dependency only. Training and replay use
the locked Python3.15/Torch2.15 runtime. Draw from the committed report JSON.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
COLORS = {"add": "#3e5266", "qk": "#d08a00", "conditioned_qk": "#00836b",
          "scaled": "#8156b6", "attnres": "#c34145"}
NAMES = {"add": "ordinary", "qk": "raw Q/K", "conditioned_qk": "conditioned Q/K",
         "scaled": "scaled addition", "attnres": "paper AttnRes"}


def main():
    report = json.loads((HERE / "results.json").read_text())
    for horizontal in ("step", "wall_seconds"):
        fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharey=True)
        for case in report["cases"]:
            if "learning_curve" not in case:
                continue
            row = (97, 193).index(case["prime"])
            column = (11, 17).index(case["seed"])
            axis = axes[row, column]
            points = case["learning_curve"]
            x = [point[horizontal] for point in points]
            color = COLORS[case["mode"]]
            axis.plot(x, [100 * p["validation"]["accuracy"] for p in points],
                      color=color, label=NAMES[case["mode"]], linewidth=1.7)
            axis.plot(x, [100 * p["train"]["accuracy"] for p in points],
                      color=color, linestyle=":", alpha=.35, linewidth=.9)
            axis.scatter(x[-1], 100 * points[-1]["validation"]["accuracy"], color=color,
                         marker="o" if case["stable_validation"] else "x", s=22)
        for row, prime in enumerate((97, 193)):
            for column, seed in enumerate((11, 17)):
                axis = axes[row, column]
                axis.set_title(f"P={prime}; initialization seed {seed}")
                axis.set_ylim(-2, 103)
                axis.axhline(99.9, color="#888888", linewidth=.6, linestyle="--")
                axis.grid(alpha=.15)
                axis.set_xlabel("Optimizer updates" if horizontal == "step" else "Run wall seconds (including setup)")
                if column == 0:
                    axis.set_ylabel("Accuracy (%)")
        axes[0, 0].legend(fontsize=8, loc="lower right")
        fig.suptitle("Residual mixing in the normalized RoPE transformer: every declared case")
        fig.text(.5, .015, "Solid: validation; dotted: train. Circle: stable validation stop; cross: update/time cap. Final test is reported separately.",
                 ha="center", fontsize=8)
        fig.tight_layout(rect=(0, .04, 1, .95))
        name = "learning_curves" if horizontal == "step" else "learning_wall_time"
        fig.savefig(HERE / f"{name}.svg")
        fig.savefig(HERE / f"{name}.png", dpi=170)
        plt.close(fig)


if __name__ == "__main__":
    main()
