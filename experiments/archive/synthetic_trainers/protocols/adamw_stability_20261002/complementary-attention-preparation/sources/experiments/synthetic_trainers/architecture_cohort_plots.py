"""Render every held-out pair and its frozen tail/recovery support."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def render_cohort(directory, result, recovery, plan):
    directory = Path(directory); output = directory / "plots"; output.mkdir(exist_ok=True)
    label = "CPU pipeline fixtures" if plan["CPU_fixture"] else "Scientific held-out comparisons"
    figures = []
    figure, axes = plt.subplots(3, 2, figsize=(13, 11), constrained_layout=True)
    for axis, pair in zip(axes.flat, plan["pairs"]):
        for role, color in (("control", "tab:blue"), ("candidate", "tab:orange")):
            name = pair[role]
            report = json.loads((directory / "runs" / name / "measurements.json").read_text())
            history = report["history"]
            axis.plot([p["step"] for p in history], [p["heldout"]["accuracy"] for p in history],
                      color=color, label=role)
            axis.plot([p["step"] for p in history], [p["train"]["accuracy"] for p in history],
                      color=color, alpha=.4, linestyle="--")
            start = report["completed_steps"] - plan["criterion"]["tail_steps"]
        axis.axvspan(start, report["completed_steps"], alpha=.08, color="gray")
        axis.axhline(plan["criterion"]["target"], color="black", linestyle=":")
        axis.set_title(pair["name"]); axis.set_ylim(-.02, 1.02); axis.set_xlabel("Completed updates")
        axis.set_ylabel("Complete RHS accuracy"); axis.legend(); axis.grid(alpha=.2)
    figure.suptitle(label + ": native AdamW architecture pairs, all full budgets and failures\n"
                    "Solid held-out / dashed train; gray band is the frozen final tail")
    figures.append((figure, "architecture-comparison"))
    figure, axes = plt.subplots(2, 1, figsize=(13, 9), constrained_layout=True)
    for pair in plan["pairs"]:
        for role, style in (("control", "-"), ("candidate", "--")):
            metrics = recovery["recovery"][pair[role]]["tail_and_episodes"]["metrics"]["joint"]
            windows = metrics["fixed_tail_windows"]
            axes[0].plot([w["start"] for w in windows], [w["failure_fraction"] for w in windows],
                         linestyle=style, marker=".", label=f"{pair['name']} {role}")
            values = [w for w in windows if w["episode_onsets_per_10000_updates"] is not None]
            if values:
                axes[1].plot([w["start"] for w in values], [w["episode_onsets_per_10000_updates"] for w in values],
                             linestyle=style, marker=".", label=f"{pair['name']} {role}")
    axes[0].set_ylabel("Joint failures / scheduled observations"); axes[0].set_ylim(-.02, 1.02)
    axes[0].legend(ncol=2, fontsize=8)
    axes[1].set_ylabel("New post-confirmation episodes per 10,000 updates")
    if not axes[1].lines:
        axes[1].text(.5, .5, "No first long joint confirmation: episode rates unavailable",
                     transform=axes[1].transAxes, ha="center", va="center")
    for axis in axes:
        axis.set_xlabel("Frozen final-window start"); axis.grid(alpha=.2)
    figure.suptitle(label + ": all paired failure-frequency outcomes\nMissing long confirmations remain unavailable, not zero")
    figures.append((figure, "architecture-recovery"))
    for figure, name in figures:
        figure.savefig(output / f"{name}.png", dpi=150)
        figure.savefig(output / f"{name}.pdf")
        plt.close(figure)
