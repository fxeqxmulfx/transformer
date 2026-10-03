"""Execute both full native AdamW budgets from the committed conditional schedule plan."""

import argparse
import json
from pathlib import Path
import subprocess

from experiments.synthetic_trainers.scheduled_protocol import run_pair,verify_live


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only",action="store_true")
    args=parser.parse_args()
    protocol=Path(__file__).resolve().parent
    manifest=protocol/"scheduled-pair-plan.json"
    subprocess.run(["git","ls-files","--error-unmatch",str(manifest)],check=True,stdout=subprocess.DEVNULL)
    subprocess.run(["git","diff","--quiet","HEAD","--",str(manifest)],check=True)
    plan=json.loads(manifest.read_text())
    stage=Path(plan["output_directory"])
    verify_live(stage,plan)
    if not plan["scientific_run"]:
        raise ValueError("Scientific launcher cannot execute a CPU pipeline fixture")
    if args.check_only:
        print("Both full native AdamW schedule cases, complete reference, sources and scoring verified.")
        return
    run_pair(stage,plan,progress=lambda row:print(json.dumps(row),flush=True))


if __name__ == "__main__":
    main()
