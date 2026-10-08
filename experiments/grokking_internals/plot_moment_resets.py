"""Plot all same-gradient moment-reset counterfactuals on archived states.

Source: moment_reset_results.json, native AdamW at 0033b1b, and
Grokking.AdamW.PartialReset. Separate reset branches keep the old clock;
the fresh branch clears it. Points are single disposable CPU steps,
not resumed trajectories, intervention schedules or causal grokking times.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
STYLES = (("retained", "Retained buffers", "#176799", "o", -.24),
          ("fresh_adamw", "Both buffers + clock reset", "#b35420", "s", -.08),
          ("reset_first_moment", "First moment reset", "#37944b", "^", .08),
          ("reset_second_moment", "Second moment reset", "#ad447e", "v", .24))


def main():
    data = json.loads((HERE / "moment_reset_results.json").read_text())
    records = [(run["label"], record["step"], record["measurement"])
               for run in data["runs"] for record in run["records"]]
    labels = {"gptmini-seed1": "S1", "gptmini-seed2": "S2", "gptmini-seed3": "S3",
              "reference-fraction20": "Ref"}
    fig, axes = plt.subplots(2, 1, figsize=(14.4, 7.4), sharex=True)
    for axis, metric in zip(axes, ("CE", "accuracy")):
        def value(population):
            return population["losses"]["answer"] if metric == "CE" else 100 * population["answer_accuracy"]
        baseline = [value(m["before"]["heldout_nonzero"]) for _, _, m in records]
        axis.scatter(range(len(records)), baseline, color="#30363c", marker="x", s=39,
                     label="Before step", zorder=4)
        for branch, name, color, marker, offset in STYLES:
            values = [value(m["branches"][branch]["populations"]["heldout_nonzero"]["after"])
                      for _, _, m in records]
            axis.scatter([i + offset for i in range(len(records))], values, color=color,
                         marker=marker, s=30, label=name, zorder=3)
        for i, (label, _, _) in enumerate(records):
            if i and records[i - 1][0] != label:
                axis.axvline(i - .5, color="#858c94", linewidth=.9, alpha=.6)
        axis.grid(alpha=.15, axis="y")
        if metric == "CE":
            axis.set_yscale("log")
            axis.set_ylabel("Held-out nonzero answer CE (log scale)")
        else:
            axis.set_ylabel("Held-out nonzero answer accuracy (%)")
            axis.set_ylim(-3, 103)
    axes[0].set_title("One same-gradient native AdamW step: buffer-reset counterfactuals on all 19 states")
    axes[1].set_xticks(range(len(records)), [f"{labels[label]}: {step / 1000:g}k"
                                            for label, step, _ in records], rotation=45, ha="right")
    axes[1].set_xlabel("Archived run and completed-update clock (independent starting states)")
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", bbox_to_anchor=(.5, .048), ncol=5, frameon=False, fontsize=9)
    fig.text(.5, .018, "Identical starting weights, clipped next-batch gradient and rate 0.001; CPU copies only; no training resumed.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .105, 1, 1))
    svg = HERE / "moment_resets.svg"
    fig.savefig(svg)
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    fig.savefig(HERE / "moment_resets.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
