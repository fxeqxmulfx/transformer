"""Exercise and archive two full-width native schedule budgets on CPU before scientific selection."""

from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess

import torch

from experiments.synthetic_trainers.paper_reproduction.diagnostics import DiagnosticsConfig
from experiments.synthetic_trainers.persistence import PersistenceConfig
from experiments.synthetic_trainers.runtime import write_json
from experiments.synthetic_trainers.scheduled_layout import current_sources,digest
from experiments.synthetic_trainers.scheduled_pair_report import save_pair
from experiments.synthetic_trainers.scheduled_protocol import freeze_pair,run_pair
from experiments.synthetic_trainers.scheduled_report import save_run
from experiments.synthetic_trainers.scheduled_training import ScheduledRunConfig


def main():
    if torch.cuda.is_initialized():
        raise ValueError("Preparation is CPU-only")
    torch.set_num_threads(1)
    protocol=Path(__file__).resolve().parent
    output=protocol/"scheduled-pair-preparation"
    stage=Path("experiments/runs/adamw_stability_20261002/bootstrap/schedule_pair_CPU_pipeline")
    if output.exists() or stage.exists():
        raise FileExistsError("CPU schedule pipeline preparation requires fresh destinations")
    base=ScheduledRunConfig(model="gptmini",optimizer="adamw",prime=7,train_fraction=.5,
        width=128,layers=2,heads=4,steps=40,batch_size=4,eval_every=5,
        learning_rate=.0003,weight_decay=.1,warmup_steps=10,device="cpu",
        anneal_start=20,anneal_end=30,final_rate_factor=.1)
    criterion=PersistenceConfig(plateau_steps=5,plateau_observations=2,confirmation_observations=2,tail_steps=10)
    plan=freeze_pair(stage,base,DiagnosticsConfig(5,True,True),criterion)
    state=run_pair(stage,plan)
    output.mkdir()
    archives=[]
    for recipe in plan["recipes"]:
        path=output/recipe["name"]
        save_run(stage,recipe["name"],path)
        archives.append(path)
    paired=output/"pair"
    result=save_pair(archives,paired)
    code="from experiments.synthetic_trainers.scheduled_pair_report import verify_pair; import sys,json; result=verify_pair(sys.argv[1]); print(json.dumps({'complete_pair':result['complete_pair'],'scientific_run':result['scientific_run'],'torch_imported':'torch' in sys.modules}))"
    offline=json.loads(subprocess.check_output(["/usr/bin/python3","-c",code,str(paired)],text=True))
    if not offline["complete_pair"] or offline["scientific_run"] or offline["torch_imported"] or torch.cuda.is_initialized():
        raise ValueError("CPU schedule pipeline did not verify portably without Torch or CUDA initialization")
    write_json(output/"validation.json",{"recorded_at_utc":datetime.now(timezone.utc).isoformat(),
        "scope":"actual CPU full-width 40-update paired fixture; no scientific learning result",
        "command":"CUDA_VISIBLE_DEVICES='' .venv/bin/python -m experiments.synthetic_trainers.protocols.adamw_stability_20261002.prepare_schedule_pair",
        "complete_cases":state["completed_runs"],"updates_per_case":40,"total_updates":80,
        "same_width_layers_heads_as_scientific_model":True,"offline_verification":offline,
        "GPU_context_initialized":False,"scientific_schedule_selected":False,
        "scientific_schedule_frozen":False,"scientific_schedule_started":False,
        "all_negative_outcomes_retained":True,"campaign_architecture_claim_allowed":False,
        "source_hashes":current_sources(),"pair_artifact_manifest_sha256":digest(paired/"artifact-hashes.json")})
    print(json.dumps({"complete_CPU_cases":2,"updates_per_case":40,"offline_verification":offline,
        "eligible_pair_support":result["eligible_pair_support"],"scientific_schedule_started":False}))


if __name__ == "__main__":
    main()
