"""Plot selected fixed-displacement observations from the complete reader.

Source: curvature_profile_results.json, pinned native CPU states and
Grokking.AdamW.CurvatureBound at cb38c97. The three panels illustrate
the positive finite training change, held-out overshoot after a negative
slope, and an incorrect initial-Hessian prediction on an early-learning
control. Every one of the 19 states remains in the JSON and README table.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent


def main():
    data = json.loads((HERE / "curvature_profile_results.json").read_text())
    rows = {(run["label"], record["step"]): record["measurement"]
            for run in data["runs"] for record in run["records"]}
    choices = (("gptmini-seed1", 30000, "train_all", "Seed 1, step 30k: full train CE"),
               ("gptmini-seed1", 36000, "heldout_nonzero", "Seed 1, step 36k: held-out answer CE"),
               ("gptmini-seed2", 35000, "train_all", "Seed 2, step 35k: full train CE"))
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.1))
    for axis, (label, step, population, title) in zip(axes, choices):
        measurement = rows[label, step]
        records = measurement["populations"][population]["records"]
        initial = records[0]
        fractions = [i / 100 for i in range(101)]
        rates = [fraction * measurement["rate"] for fraction in fractions]
        linear = [eta * initial["rate_derivative"] for eta in rates]
        quadratic = [value + eta ** 2 * initial["rate_curvature"] / 2 for value, eta in zip(linear, rates)]
        axis.axhline(0, color="#8b929a", linewidth=.8)
        axis.plot(fractions, linear, "--", color="#8f9baa", label="Initial slope")
        axis.plot(fractions, quadratic, color="#e69a32", label="Initial slope + Hessian")
        axis.plot([record["fraction"] for record in records], [record["loss_change"] for record in records],
                  "o-", markersize=4.5, color="#196a9c", label="Observed CE (7 points)")
        axis.set_title(title, fontsize=10)
        axis.set_xlabel("Fraction of fixed native CPU displacement")
        axis.set_ylabel("CE change from starting weights")
        axis.grid(alpha=.18)
        axis.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3), useMathText=True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .07), ncol=3, frameon=False, fontsize=9)
    fig.text(.5, .015, "Disposable CPU counterfactuals; sampled curvature does not certify an interval bound.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .18, 1, 1))
    fig.savefig(HERE / "curvature_profiles.svg")
    fig.savefig(HERE / "curvature_profiles.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
