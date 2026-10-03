"""Execute the committed first six-case tagged native AdamW confirmation without early stopping."""

import argparse
import json
from pathlib import Path
import subprocess

from experiments.synthetic_trainers.tagged_confirmation_protocol import run_confirmation,verify_live
from experiments.synthetic_trainers.tagged_confirmation_report import assemble,verify_confirmation


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--check-only",action="store_true")
    args=parser.parse_args()
    manifest=Path(__file__).resolve().parent/"tagged-confirmation-plan.json"
    subprocess.run(["git","ls-files","--error-unmatch",str(manifest)],check=True,stdout=subprocess.DEVNULL)
    subprocess.run(["git","diff","--quiet","HEAD","--",str(manifest)],check=True)
    plan=json.loads(manifest.read_text())
    if not plan["scientific_run"] or plan["CPU_fixture"] or Path(plan["output_directory"]).resolve()!=args.output.resolve():
        raise ValueError("The committed scientific cohort must use its own declared directory")
    verify_live(args.output,plan)
    if args.check_only:
        print("All six fresh tagged native cases, calibration, sources, corpora and scoring verified.")
        return
    run_confirmation(args.output,plan,progress=lambda row:print(json.dumps(row),flush=True))
    destination=Path("experiments/synthetic_trainers/baselines")/f"adamw_tagged_{plan['selected_recipe']}_six_case_confirmation_20261002"
    result=verify_confirmation(destination) if destination.exists() else assemble(args.output,destination)
    code="from experiments.synthetic_trainers.tagged_confirmation_report import verify_confirmation; import sys,json; r=verify_confirmation(sys.argv[1]); print(json.dumps({'completed_runs':r['completed_runs'],'torch_imported':'torch' in sys.modules}))"
    offline=json.loads(subprocess.check_output(["/usr/bin/python3","-c",code,str(destination)],text=True))
    if offline!={"completed_runs":6,"torch_imported":False}:
        raise ValueError("The complete six-case archive must verify without Torch")
    print(json.dumps({"archive":str(destination),"stable_grokking_runs":result["stable_grokking_runs"],
        "repeatable_stable_benchmark":result["repeatable_stable_benchmark"]}),flush=True)


if __name__=="__main__":
    main()
