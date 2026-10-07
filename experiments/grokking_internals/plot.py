"""Scientific plots of all six measurements from the committed snapshot.

From python/: uv run --locked --with matplotlib==3.10.8 python
../experiments/grokking_internals/plot.py. Curves are measured checkpoint
values, not fitted forecasts. No missing historical activations are filled.
"""

import json
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
plt.rcParams.update({"svg.fonttype": "none", "svg.hashsalt": "grokking_internals_v1",
                     "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name):
    fig.savefig(HERE / f"{name}.svg", metadata={"Date": None})
    path = HERE / f"{name}.svg"
    path.write_text(re.sub(r"[ \t]+(?=\n)", "", path.read_text()))
    fig.savefig(HERE / f"{name}.png", dpi=160)
    plt.close(fig)


def dashboard(run):
    rows = [row for row in run["snapshots"] if row["step"] <= 40000]
    x = [row["step"] / 1000 for row in rows]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout="constrained")
    a, b, c, d, e, f = axes.flat
    a.plot(x, [r["output"]["raw_heldout_accuracy"] for r in rows], "o-", label="Model held-out")
    for name, label in (("blocks.1.attention", "Block 1 attention"), ("blocks.1.neurons", "Block 1 neurons"),
                        ("blocks.1", "Block 1 residual")):
        a.plot(x, [r["features"][name]["linear_probe"]["heldout_accuracy"] for r in rows], ".-", label=label)
    a.set(title="Frozen ridge probes fitted only on train", ylabel="Held-out answer accuracy", ylim=(-.03, 1.03))
    a.legend(fontsize=8)
    b.plot(x, [r["gradients"]["groups"]["all"]["train"]["mean_pair_cosine"] for r in rows], "o-", label="Between train batches")
    b.plot(x, [r["gradients"]["groups"]["all"]["train_heldout_mean_gradient_cosine"] for r in rows], "o-", label="Train vs held-out mean")
    b.set(title="Gradient agreement: answer + EOS objective", ylabel="Cosine", ylim=(-1.03, 1.03))
    b.legend(fontsize=8)
    for name, label in (("blocks.0", "Block 0 residual"), ("blocks.1", "Block 1 residual"),
                        ("blocks.1.neurons", "Block 1 neurons")):
        c.plot(x, [r["features"][name]["heldout_spectrum"]["energy_entropy_rank"] for r in rows], "o-", label=label)
    c.set(title="Centered activation energy spectra", ylabel="Energy entropy effective rank")
    c.legend(fontsize=8)
    for name in ("blocks.0.neurons", "blocks.1.neurons"):
        d.plot(x, [r["neurons"][name]["exact_zero_heldout_fraction"] for r in rows], "o-", label=f"{name}: zeros")
        d.plot(x, [r["neurons"][name]["median_profile_cosine"] for r in rows], ".--", label=f"{name}: profile agreement")
    d.set(title="FFN sparsity and nonzero-quotient profiles", ylabel="Fraction / train-test profile cosine")
    d.legend(fontsize=8)
    e.plot(x, [max(v["heldout"]["accuracy_drop"] for v in r["ablations"]["heads"].values()) for r in rows], "o-", label="Largest individual-head drop")
    for name in ("top", "bottom", "random"):
        e.plot(x, [r["ablations"]["subspaces"][f"blocks.1.{name}"]["heldout"]["accuracy_drop"] for r in rows], ".-", label=f"Block 1: {name} 8 directions")
    e.set(title="Causal removals on saved model copies", ylabel="Held-out accuracy drop")
    e.legend(fontsize=8)
    changed = [r for r in rows if r["functional_update"] is not None]
    u = [r["step"] / 1000 for r in changed]
    f.plot(u, [r["functional_update"]["heldout"]["geometry_change_energy_fraction"] for r in changed], "o-", label="Geometry share of logit change")
    f.plot(u, [r["functional_update"]["heldout"]["prediction_change_fraction"] for r in changed], "o-", label="Changed predicted answers")
    f.set(title="Functional change between retained endpoints", ylabel="Fraction", ylim=(-.03, 1.03))
    f.legend(fontsize=8)
    for axis in axes.flat:
        axis.set_xlabel("Training updates (thousands)")
        axis.grid(alpha=.2)
        axis.set_xlim(0, max(40, max(x)))
        axis.axvline(35.5, color="gray", linestyle=":", linewidth=1)
    fig.suptitle("Ordinary GPTMini + AdamW: six internal measurements through update 40,000\n"
                 "Dotted line: first 99% answer accuracy in the full evaluation history; block indices start at zero", fontsize=13)
    save(fig, "six_measurements")


def control(report):
    keys = ("grokking_internals/gptmini-seed1", "grokking_progress/reference-fraction20")
    if any(key not in report["runs"] for key in keys):
        return
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout="constrained")
    for key, label in zip(keys, ("GPTMini delayed repeat", "Reference memorization control")):
        rows = report["runs"][key]["snapshots"]
        x = [row["step"] / 1000 for row in rows]
        axes[0].plot(x, [r["output"]["raw_heldout_accuracy"] for r in rows], "o", label=label)
        axes[1].plot(x, [r["features"]["blocks.1"]["linear_probe"]["heldout_accuracy"] for r in rows], "o", label=label)
        axes[2].plot(x, [r["gradients"]["groups"]["all"]["train_heldout_mean_gradient_cosine"] for r in rows], "o", label=label)
    for axis, title in zip(axes, ("Model held-out accuracy", "Frozen block 1 probe accuracy", "Train-test gradient cosine")):
        axis.set(title=title, xlabel="Training updates (thousands)")
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
    fig.suptitle("Recorded checkpoints only; control differs in architecture, data fraction and decay\n"
                 "Points are retained weights; missing intervals are not interpolated", fontsize=12)
    save(fig, "control_comparison")


if __name__ == "__main__":
    report = json.loads((HERE / "internal_suite_results.json").read_text())
    if "grokking_internals/gptmini-seed1" in report["runs"]:
        dashboard(report["runs"]["grokking_internals/gptmini-seed1"])
    control(report)
