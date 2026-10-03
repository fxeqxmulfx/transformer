"""Freeze the user-directed normalizer pair after the complete 300,000-update reference."""

from pathlib import Path
import json
import subprocess

from experiments.synthetic_trainers.attention_protocol import freeze_pair
from experiments.synthetic_trainers.attention_training import AttentionRunConfig
from experiments.synthetic_trainers.paper_reproduction.diagnostics import DiagnosticsConfig
from experiments.synthetic_trainers.persistence import PersistenceConfig
from experiments.synthetic_trainers.runtime import write_json


def main():
    protocol=Path(__file__).resolve().parent
    expected=protocol/"attention-pair-plan.json"
    if expected.exists():
        raise FileExistsError("The scientific pair must be frozen exactly once")
    reference=Path("experiments/synthetic_trainers/baselines/adamw_stability_mod193_fraction25_budget300k_calibration_20261002")
    parent=json.loads((reference/"plan.json").read_text())
    if not (protocol/"budget300k-result-validation.json").exists():
        raise ValueError("Review and commit the complete reference result before freezing sparsemax")
    validation_path=protocol/"budget300k-result-validation.json"
    validation=json.loads(validation_path.read_text())
    if (validation.get("completed_updates") != 300000 or not validation.get("offline_verification_passed")
            or not validation.get("PNG_figures_visually_reviewed") or not validation.get("PDF_figures_visually_reviewed")):
        raise ValueError("Complete reference verification and actual PNG/PDF review remain required")
    subprocess.run(["git","ls-files","--error-unmatch",str(validation_path)],check=True,stdout=subprocess.DEVNULL)
    subprocess.run(["git","diff","--quiet","HEAD","--",str(validation_path)],check=True)
    config=AttentionRunConfig(**parent["recipes"][0]["config"],attention_normalization="softmax")
    stage=Path("experiments/runs/adamw_stability_20261002/attention_mod193_fraction25_lr0003_budget300k")
    plan=freeze_pair(stage,config,DiagnosticsConfig(**parent["instrumentation"]),
                     PersistenceConfig(**parent["criterion"]),reference=reference)
    plan["output_directory"]=str(stage)
    write_json(stage/"plan.json",plan)
    write_json(expected,plan)
    print(json.dumps({"frozen_utc":plan["frozen_utc"],"stage":str(stage),"planned_runs":2,"scientific_training_started":False}))


if __name__ == "__main__":
    main()
