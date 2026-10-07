"""Reproduce causal loss-alarm evaluation on fixed archived histories.

Run from python/ with `uv run --locked python
../experiments/grokking_progress/archived.py`. This reads stored observations
only; it never trains or edits source runs. Ground truth is restricted to
the recorded budget and is used only after causal indicators are computed.
"""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from lab.domain.grokking import ProgressCriterion, progress

ROOT = Path(__file__).resolve().parents[2]
STUDY = Path(__file__).resolve().parent


def summarize(label, source, history, model, optimizer):
    evidence = progress(history)
    events = evidence["first_events"]
    delayed = events["observed_delayed_generalization"]
    alarm = events["loss_only_alarm"]
    crossing = next((p["step"] for p in history if p["heldout"]["answer_accuracy"] >= .99), None)
    confirmed = next((point["step"] for point in evidence["observations"] if point["phase"] == "generalized"), None)
    fit = next((point["step"] for point in evidence["observations"] if point["train_fit"]), None)
    return {"label": label, "model": model, "optimizer": optimizer,
            "source": str(source.relative_to(ROOT)), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "completed_updates": history[-1]["step"], "observations": len(history),
            "confirmed_train_fit": fit, "first_heldout_99": crossing,
            "confirmed_generalization": confirmed, "observed_delayed_generalization": delayed,
            "loss_only_alarm": alarm,
            "lead_to_confirmed_delayed_generalization": delayed - alarm if delayed is not None and alarm is not None else None,
            "loss_alarm_without_observed_delayed_generalization": alarm is not None and delayed is None,
            "loss_alarm_count": sum(p["loss_only_alarm"] for p in evidence["observations"]),
            "final_heldout_answer_accuracy": history[-1]["heldout"]["answer_accuracy"],
            "loss_component": evidence["loss_component"],
            "structure_observations": evidence["structure_observations"]}


def main():
    baselines = ROOT / "experiments/archive/synthetic_trainers/baselines"
    names = ["mod97_fraction20_reference", "mod97_fraction50_wd1_reference", "mod97_fraction50_wd01_reference"]
    names += [f"mod97_fraction50_wd01_confirmation_{model}_seed{seed}"
              for model in ("reference", "gptmini_adamw") for seed in (1, 2, 3)]
    rows = []
    for name in names:
        source = baselines / f"{name}_20261002/measurements.json"
        runs = json.loads(source.read_text())["runs"]
        if len(runs) != 1 or runs[0]["completed_steps"] != 150_000:
            raise ValueError("The baseline must contain one complete run")
        run = runs[0]
        config = run["plan"]["config"]
        rows.append(summarize(name, source, run["history"], config["model"], config["optimizer"]))
    for label in ("base", "sparsemax", "repair-qknorm-one", "repair-qknorm-one-seed1"):
        source = ROOT / f"experiments/mod193_stability/runs/{label}/history.jsonl"
        history = [json.loads(line) for line in source.read_text().splitlines()]
        if history[-1]["step"] != 300_000:
            raise ValueError("The mod-193 control must be complete")
        rows.append(summarize(f"mod193-{label}", source, history,
                              "gptmini-softmax" if label == "base" else "gptmini-sparsemax", "adamw"))
    report = {"criterion": asdict(ProgressCriterion()), "runs": rows,
              "scope": "archived_known_recipes; full_recorded_budgets; no_intermediate_orbit_probes",
              "false_alarm_definition": "loss_alarm_without_observed_delayed_generalization_within_recorded_budget",
              "loss_only_false_alarm_runs": sum(row["loss_alarm_without_observed_delayed_generalization"] for row in rows)}
    (STUDY / "archived_loss_baseline.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    for row in rows:
        print(row["label"], "fit", row["confirmed_train_fit"], "alarm", row["loss_only_alarm"],
              "delayed", row["observed_delayed_generalization"], "final", row["final_heldout_answer_accuracy"])


if __name__ == "__main__":
    main()
