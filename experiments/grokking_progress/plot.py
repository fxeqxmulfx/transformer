"""Standalone scientific figures from archived and current raw observations.

From python/: uv run --locked --with matplotlib==3.10.8 python
../experiments/grokking_progress/plot.py. Figures state the recorded budget;
unfinished runs remain labeled partial. No signal reads future weights.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.rcParams["svg.hashsalt"] = "grokking-progress-v1"
from matplotlib import pyplot as plt

from lab.domain.grokking import progress

STUDY = Path(__file__).resolve().parent
ROOT = STUDY.parents[1]


def axes_style(ax):
    ax.grid(alpha=.2)
    ax.set_xlabel("Optimizer updates")


def save_svg(fig, name):
    path = STUDY / name
    fig.savefig(path, metadata={"Date": None})
    path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")


def archived():
    summary = json.loads((STUDY / "archived_loss_baseline.json").read_text())
    chosen = ["mod97_fraction50_wd01_confirmation_gptmini_adamw_seed1",
              "mod97_fraction50_wd01_confirmation_gptmini_adamw_seed2", "mod193-sparsemax"]
    fig, axes = plt.subplots(3, 2, figsize=(12, 9), layout="constrained")
    for i, label in enumerate(chosen):
        item = next(row for row in summary["runs"] if row["label"] == label)
        source = ROOT / item["source"]
        rows = ([json.loads(line) for line in source.read_text().splitlines()] if source.suffix == ".jsonl"
                else json.loads(source.read_text())["runs"][0]["history"])
        steps = [row["step"] for row in rows]
        a, b = axes[i]
        for split, color in (("train", "#3572b0"), ("heldout", "#cf4c25")):
            a.plot(steps, [row[split]["answer_accuracy"] for row in rows], color=color, label=split)
            b.plot(steps, [row[split].get("answer_loss", row[split]["loss"]) for row in rows],
                   color=color, label=split)
        for ax in (a, b):
            axes_style(ax)
            alarm = item["loss_only_alarm"]
            if alarm is not None:
                ax.axvline(alarm, color="#8b4dac", linestyle="--", label="loss-only alarm")
        a.set_title(label.replace("mod97_fraction50_wd01_confirmation_", "Mod 97 "))
        a.set_ylabel("Answer accuracy")
        a.set_ylim(-.02, 1.02)
        b.set_title("Answer CE" if "answer_loss" in rows[0]["heldout"] else "Combined answer/EOS CE")
        b.set_yscale("log")
        b.set_ylabel("Cross-entropy")
        a.legend(fontsize=8)
    fig.suptitle("Loss decline alone: delayed learning, immediate learning, and a false alarm\n"
                 "Complete archived budgets; pinned causal criterion")
    save_svg(fig, "archived_loss_baseline.svg")
    plt.close(fig)


def fresh():
    paths = sorted((STUDY / "runs").glob("*/history.jsonl"))
    if not paths:
        return
    fig, axes = plt.subplots(len(paths), 3, figsize=(15, 3.8 * len(paths)), squeeze=False, layout="constrained")
    for i, path in enumerate(paths):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        probes = [row for row in rows if "grokking" in row]
        a, b, c = axes[i]
        steps = [row["step"] for row in rows]
        for split, color in (("train", "#3572b0"), ("heldout", "#cf4c25")):
            a.plot(steps, [row[split]["answer_accuracy"] for row in rows], color=color, label=split)
        updates = [row["step"] for row in probes]
        b.plot(updates, [row["grokking"]["heldout_invariant_energy_fraction"] for row in probes], color="#27854a")
        c.plot(updates, [row["grokking"]["raw_heldout_loss"] for row in probes], label="raw held-out", color="#cf4c25")
        c.plot(updates, [row["grokking"]["heldout_restricted_loss"] for row in probes], label="held-out-only projection", color="#27854a")
        a.set_title(f"{path.parent.name}: {'complete' if rows[-1]['step'] == 150_000 else 'partial'}, update {rows[-1]['step']:,}")
        a.set_ylabel("Answer accuracy")
        a.set_ylim(-.02, 1.02)
        b.set_title("Invariant fraction of held-out logit energy")
        b.set_ylim(-.02, 1.02)
        c.set_title("Raw vs projected held-out answer CE")
        c.set_yscale("log")
        for ax in (a, b, c):
            axes_style(ax)
            event = progress(rows)["first_events"]["structure_forming"]
            if event is not None:
                ax.axvline(event, color="#8b4dac", linestyle="--")
        a.legend()
        c.legend()
    fig.suptitle("Ordinary softmax/AdamW: current functional structure and correctness\n"
                 "No training logits pooled into the main projection; no future checkpoint selection")
    save_svg(fig, "fresh_progress.svg")
    plt.close(fig)


def internals():
    path = STUDY / "runs/gptmini-seed1/internal_checkpoint_probes.jsonl"
    if not path.exists():
        return
    snapshots = [json.loads(line) for line in path.read_text().splitlines()]
    if len(snapshots) < 2:
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained")
    steps = [row["step"] for row in snapshots]
    a, b, c = axes
    for name in snapshots[0]["internal"]["features"]:
        a.plot(steps, [row["internal"]["features"][name]["invariant_energy_fraction"] for row in snapshots],
               marker="o", label=name)
    for group in ("all", "embedding", "attention", "ffn"):
        first = snapshots[0]["norms"]["groups"][group]["parameter_l2"]
        b.plot(steps, [row["norms"]["groups"][group]["parameter_l2"] / first for row in snapshots],
               marker="o", label=group)
    for name, value in snapshots[0]["internal"]["attention"].items():
        for head in range(len(value["entropy_by_head"])):
            c.plot(steps, [row["internal"]["attention"][name]["entropy_by_head"][head] for row in snapshots],
                   marker="o", label=f"{name} head {head}")
    a.set_title("Held-out invariant energy inside blocks")
    a.set_ylim(0, 1)
    b.set_title("Weight norm / norm at first snapshot")
    c.set_title("Attention entropy at answer query (nats)")
    for ax in axes:
        axes_style(ax)
        ax.legend(fontsize=7)
    fig.suptitle(f"GPTMini softmax/AdamW: internal checkpoint observations, {steps[0]:,}–{steps[-1]:,}\n"
                 "Actual saved CUDA-trained weights observed on CPU; lines connect measured points")
    save_svg(fig, "internal_progress.svg")
    fig.savefig(STUDY / "internal_progress.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    archived()
    fresh()
    internals()
