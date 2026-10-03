"""Conditionally freeze scheduling after both normalizer results are reviewed and committed."""

import json
from pathlib import Path
import subprocess

from experiments.synthetic_trainers.attention_report import verify_pair
from experiments.synthetic_trainers.paper_reproduction.diagnostics import DiagnosticsConfig
from experiments.synthetic_trainers.persistence import PersistenceConfig
from experiments.synthetic_trainers.runtime import write_json
from experiments.synthetic_trainers.scheduled_protocol import freeze_pair
from experiments.synthetic_trainers.scheduled_training import ScheduledRunConfig


def main():
    protocol=Path(__file__).resolve().parent
    expected=protocol/"scheduled-pair-plan.json"
    if expected.exists():
        raise FileExistsError("Scientific schedule calibration must be frozen exactly once")
    validation_path=protocol/"attention-pair-result-validation.json"
    if not validation_path.exists():
        raise ValueError("Complete, review and commit both normalizer runs before selecting scheduling")
    validation=json.loads(validation_path.read_text())
    if (validation.get("completed_updates_per_run") != 300000
            or validation.get("completed_scientific_runs") != 2
            or any(validation.get(key) is not True for key in (
                "offline_verification_passed","PNG_figures_visually_reviewed",
                "PDF_figures_visually_reviewed","trainer_and_archive_worker_terminal_sessions_consumed"))):
        raise ValueError("Both full normalizer budgets, portable verification, figure review and terminal sessions are required")
    subprocess.run(["git","ls-files","--error-unmatch",str(validation_path)],check=True,stdout=subprocess.DEVNULL)
    subprocess.run(["git","diff","--quiet","HEAD","--",str(validation_path)],check=True)
    reference=Path("experiments/synthetic_trainers/baselines/adamw_attention_softmax_sparsemax_mod193_fraction25_lr0003_budget300k_pair_20261002")
    result=verify_pair(reference)
    if result["outcome"]["control"]["stable_grokking"]:
        raise ValueError("The passing primary softmax recipe must enter independent confirmation")
    parent=json.loads((reference/"plan.json").read_text())
    config=dict(parent["recipes"][0]["config"])
    if config.pop("attention_normalization") != "softmax":
        raise ValueError("Scheduling retains the original softmax control")
    base=ScheduledRunConfig(**config,learning_rate_schedule="constant",
                            anneal_start=150000,anneal_end=250000,final_rate_factor=.1)
    stage=Path("experiments/runs/adamw_stability_20261002/schedule_mod193_fraction25_lr0003_budget300k")
    plan=freeze_pair(stage,base,DiagnosticsConfig(**parent["instrumentation"]),
                     PersistenceConfig(**parent["criterion"]),reference=reference)
    plan["output_directory"]=str(stage)
    write_json(stage/"plan.json",plan)
    write_json(expected,plan)
    print(json.dumps({"frozen_utc":plan["frozen_utc"],"stage":str(stage),"planned_runs":2,
                      "scientific_training_started":False,"campaign_architecture_claim_allowed":False}))


if __name__ == "__main__":
    main()
