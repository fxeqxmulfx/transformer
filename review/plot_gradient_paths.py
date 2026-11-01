"""Publication-style export of the frozen task151 curves; system Matplotlib only."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parents[1] / "experiments/residual_gradient_paths"
data = json.loads((HERE / "results.json").read_text())
fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), sharex=True)
for ax, scale in zip(axes[0], ("0.001", "1", "1000")):
    for row in data["cases"]:
        label = row["label"]
        if not label.startswith("gate-") or f"-a{scale}-" not in label:
            continue
        before, deep = "-before-" in label, "-d8-" in label
        seed = label.endswith("s11")
        history = row["history"]
        ax.plot([p["step"] for p in history],
                [max(p["validation"]["relative_mse"], 1e-16) for p in history],
                color="#b4433c" if before else "#167d9a", linestyle="--" if deep else "-",
                alpha=1 if seed else .45, linewidth=1.5,
                label=f"{'before' if before else 'after'} RMS, {'8 branches' if deep else '1 branch'}" if seed else None)
    ax.set_title(f"Gate placement: branch scale {scale}")
    ax.set_ylim(1e-15, 2)
    ax.legend(fontsize=8, loc="upper right")
for ax, depth in zip(axes[1], (1, 8, 32)):
    for row in data["cases"]:
        label = row["label"]
        if not label.startswith(f"transport-d{depth}-"):
            continue
        ordinary, seed = "-ordinary-" in label, label.endswith("s11")
        history = row["history"]
        ax.plot([p["step"] for p in history],
                [max(p["validation"]["relative_mse"], 1e-16) for p in history],
                color="#b4433c" if ordinary else "#167d9a", linestyle="--" if ordinary else "-",
                alpha=1 if seed else .45, linewidth=1.5,
                label=("branch -h/2" if ordinary else "branch -h/(2D)") if seed else None)
    ax.set_title(f"Residual transport: depth {depth}")
    ax.set_ylim(1e-14, 20)
    ax.set_xlabel("Optimizer updates")
    ax.legend(fontsize=8, loc="upper right")
for ax in axes.flat:
    ax.set_yscale("log")
    ax.axhline(1e-4, color="#6c7280", linewidth=.7, linestyle=":")
    ax.grid(alpha=.18)
    ax.set_xlim(0, 10000)
axes[0, 0].set_ylabel("Validation relative MSE")
axes[1, 0].set_ylabel("Validation relative MSE")
fig.suptitle("Known solutions, matched initial outputs, different gradient paths", fontsize=15)
fig.text(.5, .015, "36 retained CPU cases • Schedule-Free AdamW • two data/batch seeds • supplied features, not a grokking test",
         ha="center", fontsize=9)
fig.tight_layout(rect=(0, .035, 1, .95))
fig.savefig(HERE / "learning_curves.png", dpi=180)
fig.savefig(HERE / "learning_curves.svg")
