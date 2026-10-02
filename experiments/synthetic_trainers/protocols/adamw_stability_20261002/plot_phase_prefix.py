"""Plot the verified partial phase history in a fresh standalone destination."""

import argparse
import hashlib
import json
from pathlib import Path

from .verify_phase_prefix import verify


def render(source, destination):
    verification = verify(source)
    destination.mkdir()
    history = [json.loads(line) for line in (source / "history.jsonl").read_text().splitlines()]
    summary = json.loads((source / "summary.json").read_text())
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    xs = [point["step"] for point in history]
    figure, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True, constrained_layout=True)
    for split, color in (("train", "#0072b2"), ("heldout", "#d55e00")):
        axes[0].plot(xs, [point[split]["accuracy"] for point in history],
                     marker=".", markersize=3, label=split, color=color)
    axes[0].axhline(summary["criterion"]["target"], color="black", linestyle=":", linewidth=.8,
                   label="99% target")
    axes[0].set(ylim=(-.02, 1.025), ylabel="Complete RHS accuracy")
    for metric, color in (("answer_loss", "#d55e00"), ("EOS_loss", "#009e73")):
        axes[1].plot(xs, [max(1e-12, point["heldout"][metric]) for point in history],
                     label=metric, color=color)
    axes[1].set(yscale="log", ylabel="Held-out CE (nats; display floor 1e-12)",
                xlabel="Updates (thousands)", xlim=(0, verification["through_update"]))
    plateau, event = verification["plateau"], verification["long_confirmation"]
    for axis in axes:
        axis.axvspan(plateau["start"], plateau["end"], color="#0072b2", alpha=.10,
                     label="Measured memorization plateau")
        axis.axvspan(event["onset"], event["confirmed"], color="#009e73", alpha=.10,
                     label="20 joint target observations")
        axis.xaxis.set_major_formatter(FuncFormatter(lambda value, position: f"{value / 1000:g}"))
        axis.grid(alpha=.2)
        axis.legend(fontsize=8, loc="best")
    config = summary["config"]
    figure.suptitle(f"AdamW / GPTMini, mod {config['prime']}, {config['train_fraction']:.0%} train, "
                   f"lr {config['learning_rate']:g}, seeds {config['seed']}/{config['data_seed']}\n"
                   f"All {len(history)} canonical observations through update {verification['through_update']:,}\n"
                   "Partial calibration: final-tail persistence and independent repeatability unresolved",
                   fontsize=11)
    for extension in ("png", "pdf"):
        figure.savefig(destination / f"phase-prefix.{extension}", dpi=180, bbox_inches="tight")
    plt.close(figure)
    provenance = {"source": str(source), "source_artifact_manifest_sha256":
                  hashlib.sha256((source / "artifact-hashes.json").read_bytes()).hexdigest(),
                  "renderer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "verification": verification, "matplotlib": matplotlib.__version__,
                  "canonical_observations": len(history), "all_points_included": True,
                  "smoothing": False, "scope": "partial_ordered_phase_history; no_persistence_or_repeatability_claim"}
    (destination / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(destination.iterdir()) if p.is_file()}
    (destination / "artifact-hashes.json").write_text(json.dumps({"files": hashes}, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    render(args.source, args.destination)
