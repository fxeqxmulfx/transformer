"""Plot the complete observed cleanup certificate study without forecasts.

From python/: uv run --locked --with matplotlib==3.10.8 python
../experiments/grokking_internals/plot_geometry.py. All inputs are immutable
geometry_all_results.json measurements. Lines join available weights only;
the reference's absent early archive is left as a visible gap.
"""

import json
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

HERE = Path(__file__).resolve().parent
plt.rcParams.update({"svg.fonttype": "none", "svg.hashsalt": "grokking_geometry_v1",
                     "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
COLORS = ("#263c56", "#a04e18", "#168578")
KEYS = ("raw_nonzero_accuracy", "reference_nonzero_accuracy", "certified_fraction")
LABELS = ("Actual answers", "Cell-mean answers", "Exact certified subset")


def segments(rows, cadence):
    groups, current = [], []
    for row in rows:
        if current and row["step"] - current[-1]["step"] > cadence:
            groups.append(current)
            current = []
        current.append(row)
    if current:
        groups.append(current)
    return groups


def answers(axis, run, title, limits, cadence):
    rows = [row for row in run["records"] if limits[0] * 1000 <= row["step"] <= limits[1] * 1000]
    for key, label, color in zip(KEYS, LABELS, COLORS):
        for index, group in enumerate(segments(rows, cadence)):
            axis.plot([r["step"] / 1000 for r in group], [r["measurement"][key] for r in group],
                      marker="o", markersize=2.8, linewidth=1.2, color=color,
                      label=label if index == 0 else None, alpha=.9)
    axis.set(title=title, xlim=limits, ylim=(-.03, 1.03), ylabel="Held-out nonzero answers")
    axis.yaxis.set_major_formatter(PercentFormatter(1))
    axis.legend(fontsize=8, loc="lower right")


def main():
    data = json.loads((HERE / "geometry_all_results.json").read_text())
    runs = {f"{run['study']}/{run['label']}": run for run in data["runs"]}
    repeat = runs["grokking_internals/gptmini-seed1"]
    seed2 = runs["grokking_progress/gptmini-seed2"]
    seed3 = runs["grokking_progress/gptmini-seed3"]
    reference = runs["grokking_progress/reference-fraction20"]
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.7), layout="constrained")
    a, b, c, d, e, f = axes.flat
    answers(a, repeat, "Seed 1: full 150,000-update budget", (0, 150), 1000)
    answers(b, repeat, "Seed 1: reference precedes stable decisions", (28, 42), 1000)
    answers(c, seed2, "Seed 2: early success, later collapse", (0, 150), 5000)
    answers(d, seed3, "Seed 3: conservative bound can fall at 100%", (0, 150), 5000)
    answers(e, reference, "Reference: no certified answers", (0, 150), 5000)
    b.axvline(34, color=COLORS[1], linestyle=":", linewidth=1)
    b.axvline(35.5, color=COLORS[0], linestyle=":", linewidth=1)
    b.axvline(38, color=COLORS[2], linestyle=":", linewidth=1)
    for run, label, color, cadence in ((repeat, "Seed 1", "#263c56", 1000),
            (seed2, "Seed 2", "#a04e18", 5000), (seed3, "Seed 3", "#168578", 5000),
            (reference, "Reference", "#855ca2", 5000)):
        for index, group in enumerate(segments(run["records"], cadence)):
            f.plot([r["step"] / 1000 for r in group],
                   [r["measurement"]["mean_signed_safety"] for r in group],
                   ".-", color=color, linewidth=1, label=label if index == 0 else None)
    f.set(title="Signed margin / error balance is not monotone", xlim=(0, 150),
          ylim=(-1.03, 1.03), ylabel="Mean signed safety (float64)")
    f.legend(fontsize=8, loc="lower right")
    for axis in axes.flat:
        axis.set_xlabel("Optimizer updates (thousands)")
        axis.grid(alpha=.18)
    fig.suptitle("Ordinary softmax GPTMini + native AdamW: current-logit cleanup certificates\n"
                 "238 preserved weights; exact integer certificate counts; nonzero quotients only", fontsize=14)
    fig.supxlabel("Markers are preserved weights; connecting lines are guides. Certificates are sufficient for current answers.\n"
                  "Zoom lines: first 99% cell-mean accuracy (34k), canonical raw accuracy (35.5k), certified subset (38k).", fontsize=9)
    fig.savefig(HERE / "geometry_certificates.svg", metadata={"Date": None})
    svg = HERE / "geometry_certificates.svg"
    svg.write_text(re.sub(r"[ \t]+(?=\n)", "", svg.read_text()))
    fig.savefig(HERE / "geometry_certificates.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
