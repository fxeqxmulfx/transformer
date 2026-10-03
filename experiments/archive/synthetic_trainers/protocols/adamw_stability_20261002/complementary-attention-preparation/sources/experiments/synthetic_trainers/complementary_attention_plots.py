"""Standalone CPU preparation curves with separate novel-ID and length panels."""

import json
from pathlib import Path


def render(directory):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    directory = Path(directory)
    plan = json.loads((directory / "plan.json").read_text())
    if plan["CPU_fixture"] is not True or plan["scientific_run"] is not False:
        raise ValueError("These figures belong only to the recorded CPU preparation")
    output = directory / "plots"; output.mkdir()
    for field, title in (("ID", "Novel ID validation and teacher-forced train"),
                         ("length", "Novel longer-input validation")):
        figure, axes = plt.subplots(2, 4, figsize=(14, 7), sharex=True, sharey=True)
        for axis, pair in zip(axes.flat, plan["pairs"]):
            for norm, color in (("softmax", "tab:blue"), ("sparsemax", "tab:orange")):
                path = directory / "runs" / f"{pair['name']}-{norm}"
                history = [json.loads(line) for line in (path / "history.jsonl").read_text().splitlines()]
                steps = [row["step"] for row in history]
                if field == "ID":
                    axis.plot(steps, [row["validation_novel"]["sequence_accuracy"] for row in history],
                              color=color, marker="o", label=f"{norm}: novel ID")
                    axis.plot(steps, [row["train"]["sequence_accuracy"] for row in history],
                              color=color, linestyle=":", label=f"{norm}: train (teacher forced)")
                else:
                    for probe in history[0]["validation_ood_novel"]:
                        axis.plot(steps, [row["validation_ood_novel"][probe]["sequence_accuracy"] for row in history],
                                  color=color, marker="o", label=f"{norm}: {probe}")
            axis.set_title(pair["name"]); axis.set_ylim(-.03, 1.03)
            axis.grid(alpha=.2); axis.legend(fontsize=7)
            axis.set_xlabel("Completed updates"); axis.set_ylabel("Complete-sequence accuracy")
        figure.suptitle(f"CPU pipeline fixtures — {title}\nNo scientific learning or architecture claim")
        figure.tight_layout()
        for extension in ("png", "pdf"):
            figure.savefig(output / f"complementary-{field}.{extension}", dpi=150)
        plt.close(figure)
