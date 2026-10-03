"""Plot the actual partial constant-control phase and first sampled recovery."""

import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot(directory):
    directory = Path(directory)
    history = [json.loads(line) for line in (directory / "canonical-prefix.jsonl").read_text().splitlines()]
    triplets = json.loads((directory / "boundary-triplets.json").read_text())
    assessment = json.loads((directory / "assessment.json").read_text())
    paths = [directory / f"constant-phase-and-recovery.{extension}" for extension in ("png", "pdf")]
    if any(path.exists() for path in paths):
        raise FileExistsError("The partial scientific figures require fresh destinations")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for metric, color in (("train", "#3569b0"), ("heldout", "#dc7232")):
        axes[0].plot([p["step"] / 1000 for p in history],
                     [100 * p[metric]["accuracy"] for p in history], color=color, label=metric)
    plateau, event = assessment["plateau"], assessment["long_confirmation"]
    axes[0].axvspan(plateau["start"] / 1000, plateau["end"] / 1000, color="#999999", alpha=.15,
                   label="memorization plateau")
    axes[0].axvspan(event["onset"] / 1000, event["confirmed"] / 1000, color="#62a665", alpha=.25,
                   label="20 joint target checks")
    axes[0].set(title="Memorization and delayed generalization", ylabel="Complete RHS accuracy (%)",
                ylim=(-1, 102), xlim=(0, 106))
    axes[0].legend(loc="lower right", fontsize=8)
    neighbors = [p for p in triplets["neighbors"] if p["step"] > 105000]
    canonical = [p for p in history if p["step"] >= 105000]
    points = sorted([*canonical, *neighbors], key=lambda point: point["step"])
    for metric, color in (("train", "#3569b0"), ("heldout", "#dc7232")):
        axes[1].plot([p["step"] / 1000 for p in points], [100 * p[metric]["accuracy"] for p in points],
                     color=color, linewidth=1, label=metric)
    axes[1].scatter([p["step"] / 1000 for p in canonical],
                    [100 * p["heldout"]["accuracy"] for p in canonical],
                    marker="o", s=35, color="#dc7232", label="canonical held-out checks", zorder=3)
    axes[1].scatter([p["step"] / 1000 for p in neighbors],
                    [100 * p["heldout"]["accuracy"] for p in neighbors],
                    marker="x", s=45, color="#683b8a", label="adjacent-update probes", zorder=4)
    axes[1].set(title="First sampled failure and recovery", ylabel="Complete RHS accuracy (%)",
                ylim=(96.9, 100.2))
    axes[1].legend(loc="lower right", fontsize=8)
    for axis in axes:
        axis.axhline(99, color="#555555", linestyle="--", linewidth=.9)
        axis.set_xlabel("Completed updates (thousands)")
        axis.grid(alpha=.2)
    fig.suptitle("Native AdamW, constant rate 0.0003: partial 105,750 / 300,000 updates", fontsize=12)
    fig.text(.5, .015, "Same seeds and split as the reference; final-window persistence and independent confirmation remain unmeasured.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .04, 1, .94))
    for path in paths:
        fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    assert "torch" not in sys.modules
    return [str(path) for path in paths]


if __name__ == "__main__":
    print(json.dumps({"figures": plot(sys.argv[1]), "Torch_imported": False}))
