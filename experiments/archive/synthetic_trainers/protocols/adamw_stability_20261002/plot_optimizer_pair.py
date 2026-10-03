"""Readable standalone curves from the verified frozen optimizer pair."""

import hashlib
import json
from pathlib import Path

from experiments.synthetic_trainers.stability_comparison import verify_comparison
from .csv_verification import linear_csv_verification


def render():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    protocol = Path(__file__).resolve().parent
    source = protocol.parents[1] / "baselines/adamw_stability_optimizer_pair_lr001_20261002"
    destination = protocol / "optimizer-pair-curves"
    if destination.exists():
        raise FileExistsError("Standalone pair curves require a fresh destination")
    with linear_csv_verification():
        summary = verify_comparison(source)
    plt.rcParams.update({"font.size": 11})
    figure, axes = plt.subplots(3, 2, figsize=(15, 13), constrained_layout=True)
    inputs, counts = {}, {}
    for column, row in enumerate(summary["rows"]):
        path = source / "runs" / row["name"] / "measurements.json"
        report = json.loads(path.read_text())
        history = report["history"]
        inputs[row["name"]] = hashlib.sha256(path.read_bytes()).hexdigest()
        counts[row["name"]] = len(history)
        updates = [p["step"] / 1000 for p in history]
        for split in ("train", "heldout"):
            axes[0, column].plot(updates, [p[split]["accuracy"] for p in history], label=split)
        for metric, color in (("answer_loss", "tab:orange"), ("EOS_loss", "tab:green")):
            axes[1, column].plot(updates, [max(p["heldout"][metric], 1e-12) for p in history],
                                 label=metric, color=color)
        axes[1, column].set_yscale("log")
        axes[1, column].set_ylabel("Held-out CE (nats; display floor 1e-12)")
        for index in (0, 1):
            axes[index, column].set_xlim(0, row["steps"] / 1000)
            axes[index, column].set_xticks([0, 25, 50, 75, 100, 125, 150])
            axes[index, column].set_xlabel("Updates (thousands)")
            tail = row["steps"] - summary["criterion"]["tail_steps"]
            axes[index, column].axvspan(tail / 1000, row["steps"] / 1000, color="gray", alpha=.08)
        axes[2, column].plot([p["training_seconds"] for p in history],
                             [p["heldout"]["accuracy"] for p in history], color="tab:orange")
        axes[2, column].set_xlabel("Measured training seconds")
        axes[2, column].set_xticks([0, 1000, 2000, 3000, 4000])
        for index in (0, 2):
            axes[index, column].set_ylim(-.02, 1.02)
            axes[index, column].set_ylabel("Complete RHS accuracy")
            axes[index, column].axhline(summary["criterion"]["target"], color="black", linestyle=":", alpha=.6)
        for index in range(3):
            axes[index, column].grid(alpha=.2)
        axes[0, column].set_title("AdamW" if row["optimizer"] == "adamw" else "Raw AMSGradW")
        axes[0, column].legend(loc="lower right")
        axes[1, column].legend()
    figure.suptitle("Complete optimizer calibration: one common split and initialization\n"
                   "All scheduled observations; both phase and persistence gates fail", fontsize=16)
    destination.mkdir()
    for extension in ("png", "pdf"):
        figure.savefig(destination / f"optimizer-pair.{extension}", dpi=160)
    plt.close(figure)
    metadata = {"source_archive": str(source), "source_measurement_sha256": inputs,
                "canonical_observations_rendered": counts,
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "scope": "same_complete_curves; update_labels_in_thousands; no_change_to_frozen_archives"}
    (destination / "provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in destination.iterdir() if p.is_file()}
    (destination / "artifact-hashes.json").write_text(json.dumps({"files": hashes}, indent=2) + "\n")


if __name__ == "__main__":
    render()
