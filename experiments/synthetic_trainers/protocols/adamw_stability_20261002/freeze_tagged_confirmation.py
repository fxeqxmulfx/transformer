"""Freeze the first six fresh primary confirmations only after a committed reviewed passing pair."""

import argparse
import json
from pathlib import Path
import subprocess

from experiments.synthetic_trainers.runtime import write_json
from experiments.synthetic_trainers.tagged_confirmation_protocol import freeze


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration",type=Path,required=True)
    parser.add_argument("--recipe",required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    protocol=Path(__file__).resolve().parent
    manifest=protocol/"tagged-confirmation-plan.json"
    if manifest.exists():
        raise FileExistsError("This first scientific cohort is frozen once; consumed seeds need a new prospective cohort")
    origin=json.loads((args.calibration/"plan.json").read_text())
    if not origin["scientific_run"]:
        raise ValueError("Scientific confirmation cannot use a CPU pipeline fixture")
    if origin["stage"]=="exploratory_attention_normalizer_pair":
        validation_path=protocol/"attention-pair-result-validation.json"
    elif origin["stage"]=="conditional_fixed_schedule_calibration_pair":
        validation_path=protocol/"scheduled-pair-result-validation.json"
    else:
        raise ValueError("A complete tagged native AdamW calibration pair is required")
    if not validation_path.exists():
        raise ValueError("Complete, verify, review and commit both calibration runs before confirmation selection")
    validation=json.loads(validation_path.read_text())
    if (validation.get("completed_updates_per_run")!=300000 or validation.get("completed_scientific_runs")!=2
            or any(validation.get(key) is not True for key in ("offline_verification_passed",
                "PNG_figures_visually_reviewed","PDF_figures_visually_reviewed",
                "trainer_and_archive_worker_terminal_sessions_consumed"))):
        raise ValueError("Complete budgets, portable verification, actual figures and consumed terminal sessions are required")
    subprocess.run(["git","ls-files","--error-unmatch",str(validation_path)],check=True,stdout=subprocess.DEVNULL)
    subprocess.run(["git","diff","--quiet","HEAD","--",str(validation_path)],check=True)
    plan=freeze(args.output,args.calibration,args.recipe)
    write_json(manifest,plan)
    print(json.dumps({"planned_runs":6,"model_seeds":[4,5,6],"data_seeds":[2,3],
                      "scientific_confirmation_started":False,"output":str(args.output)}))


if __name__=="__main__":
    main()
