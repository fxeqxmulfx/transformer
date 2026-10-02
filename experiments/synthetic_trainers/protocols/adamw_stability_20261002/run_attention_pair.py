"""Execute the committed user-directed softmax/sparsemax full-budget pair."""

import argparse
import json
from pathlib import Path

from experiments.synthetic_trainers.attention_protocol import run_pair, verify_live


def read_frozen():
    protocol=Path(__file__).resolve().parent
    plan=json.loads((protocol/"attention-pair-plan.json").read_text())
    stage=Path(plan["output_directory"])
    verify_live(stage,plan)
    if not plan["scientific_run"]:
        raise ValueError("Scientific launcher cannot execute a CPU pipeline fixture")
    return stage,plan


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only",action="store_true")
    args=parser.parse_args()
    stage,plan=read_frozen()
    if args.check_only:
        print("Two full native AdamW normalizer cases, committed plan, reference, sources and scoring verified.")
        return
    run_pair(stage,plan,progress=lambda row:print(json.dumps(row),flush=True))


if __name__ == "__main__":
    main()
