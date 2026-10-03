"""Archive each complete normalizer case and the full pair without importing Torch."""

import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments.synthetic_trainers.attention_layout import current_sources,digest,validate_pair_plan
from experiments.synthetic_trainers.attention_report import save_pair,verify_pair
from experiments.synthetic_trainers.stability_report import save_run,verify_archive


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trainer-pid",type=int,required=True)
    args=parser.parse_args()
    protocol=Path(__file__).resolve().parent
    plan=json.loads((protocol/"attention-pair-plan.json").read_text())
    stage=Path(plan["output_directory"])
    metadata=stage.parent/"bootstrap/attention-pair-archive-worker.json"
    root=Path("experiments/synthetic_trainers/baselines")

    def check():
        validate_pair_plan(plan)
        if json.loads((stage/"plan.json").read_text()) != plan or current_sources() != plan["source_hashes"]:
            raise ValueError("Frozen normalizer manifest or source fingerprints changed")
        if "torch" in sys.modules:
            raise RuntimeError("Portable archive worker must not import Torch")

    def record(status,**fields):
        value={"recorded_at_utc":datetime.now(timezone.utc).isoformat(),"pid":os.getpid(),
               "trainer_pid":args.trainer_pid,"status":status,"torch_imported":"torch" in sys.modules,
               "worker_sha256":digest(__file__),**fields}
        temporary=metadata.with_suffix(".tmp")
        temporary.write_text(json.dumps(value,indent=2)+"\n");temporary.replace(metadata)
        print(json.dumps(value),flush=True)

    def offline(directory,paired=False):
        module,method=("attention_report","verify_pair") if paired else ("stability_report","verify_archive")
        code=f"from experiments.synthetic_trainers.{module} import {method}; import sys,json; {method}(sys.argv[1]); print(json.dumps({{'verified':True,'torch_imported':'torch' in sys.modules}}))"
        result=json.loads(subprocess.check_output(["/usr/bin/python3","-c",code,str(directory)],text=True))
        if not result["verified"] or result["torch_imported"]:
            raise ValueError("Complete pair must verify without Torch")

    try:
        check()
        command=Path(f"/proc/{args.trainer_pid}/cmdline").read_bytes().split(b"\0")
        if b"experiments.synthetic_trainers.protocols.adamw_stability_20261002.run_attention_pair" not in command:
            raise ValueError("The supplied process is not the committed normalizer trainer")
        paths=[]
        for index,name in enumerate(plan["execution_order"]):
            record("waiting_for_complete_normalizer_case",name=name,completed_archives=len(paths))
            while True:
                state_path=stage/"state.json"
                state=json.loads(state_path.read_text()) if state_path.exists() else {"status":"not_started","completed_runs":0}
                if state.get("completed_runs",0) > index and (stage/name/"measurements.json").exists():
                    break
                if state["status"] == "failed":
                    raise RuntimeError("Normalizer trainer failed; retain every partial history")
                if not Path(f"/proc/{args.trainer_pid}/cmdline").exists():
                    raise RuntimeError("Normalizer trainer disappeared before the case completed")
                time.sleep(40)
            check()
            destination=root/f"adamw_attention_{name}_mod193_fraction25_lr0003_budget300k_seed0_data0_20261002"
            if destination.exists():
                verify_archive(destination)
                if (destination/"measurements.json").read_bytes() != (stage/name/"measurements.json").read_bytes():
                    raise ValueError("Existing complete case archive differs from its unchanged source")
            else:
                save_run(stage,name,destination)
            offline(destination)
            paths.append(destination)
            record("normalizer_case_archive_ready_for_review_and_commit",name=name,archive=str(destination))
        check()
        pair=root/"adamw_attention_softmax_sparsemax_mod193_fraction25_lr0003_budget300k_pair_20261002"
        result=verify_pair(pair) if pair.exists() else save_pair(paths,pair,reference=stage/"reference")
        offline(pair,paired=True)
        record("complete_normalizer_pair_ready_for_review_and_commit",pair=str(pair),
               complete_pair=result["complete_pair"],eligible_pair_support=result["outcome"]["eligible_pair_support"],
               campaign_architecture_claim_allowed=False)
    except BaseException as error:
        record("failed",error=repr(error))
        raise


if __name__ == "__main__":
    main()
