"""Exercise both six-case tagged native confirmation paths on CPU; no scientific selection."""

from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess

import torch

from experiments.synthetic_trainers.runtime import write_json
from experiments.synthetic_trainers.tagged_confirmation_layout import current_sources,digest
from experiments.synthetic_trainers.tagged_confirmation_protocol import freeze,run_confirmation
from experiments.synthetic_trainers.tagged_confirmation_report import assemble


def main():
    if torch.cuda.is_initialized():
        raise ValueError("Tagged preparation is CPU-only")
    torch.set_num_threads(1)
    protocol=Path(__file__).resolve().parent
    output=protocol/"tagged-confirmation-preparation"
    if output.exists():
        raise FileExistsError("Tagged CPU preparation requires fresh outputs")
    rows=[]
    for mode,reference,name in (("normalizer","attention-pair-preparation/pair","adamw-softmax"),
                                 ("schedule","scheduled-pair-preparation/pair","adamw-cosine-tail")):
        stage=Path("experiments/runs/adamw_stability_20261002/bootstrap")/f"tagged_{mode}_six_case_CPU_pipeline"
        plan=freeze(stage,protocol/reference,name,CPU_fixture=True,render=False)
        state=run_confirmation(stage,plan)
        destination=output/mode;result=assemble(stage,destination)
        code="from experiments.synthetic_trainers.tagged_confirmation_report import verify_confirmation; import sys,json; r=verify_confirmation(sys.argv[1]); print(json.dumps({'completed_runs':r['completed_runs'],'scientific_run':r['scientific_run'],'repeatable_stable_benchmark':r['repeatable_stable_benchmark'],'torch_imported':'torch' in sys.modules}))"
        offline=json.loads(subprocess.check_output(["/usr/bin/python3","-c",code,str(destination)],text=True))
        if offline!={"completed_runs":6,"scientific_run":False,"repeatable_stable_benchmark":False,"torch_imported":False}:
            raise ValueError("Both six-case CPU fixtures must retain failures without scientific eligibility")
        rows.append({"training_mode":mode,"completed_CPU_cases":state["completed_runs"],
            "updates_per_case":plan["recipes"][0]["config"]["steps"],"total_updates":result["total_updates"],
            "offline_verification":offline,"source_hashes":plan["source_hashes"],
            "archive_artifact_manifest_sha256":digest(destination/"artifact-hashes.json")})
    if torch.cuda.is_initialized():
        raise ValueError("CPU preparation initialized CUDA")
    write_json(output/"validation.json",{"recorded_at_utc":datetime.now(timezone.utc).isoformat(),
        "scope":"two actual six-case CPU pipeline fixtures; no scientific learning result",
        "rows":rows,"all_62_sources_match":all(row["source_hashes"]==current_sources() for row in rows),
        "scientific_confirmation_selected":False,"scientific_confirmation_frozen":False,
        "scientific_confirmation_started":False,"GPU_context_initialized":False})
    print(json.dumps({"actual_CPU_cases":12,"total_updates":sum(row["total_updates"] for row in rows),
        "offline_verified_without_Torch":True,"scientific_confirmation_started":False}))


if __name__=="__main__":
    main()
