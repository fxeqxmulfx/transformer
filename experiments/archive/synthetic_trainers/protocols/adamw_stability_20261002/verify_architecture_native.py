"""Verify complete real CPU normalizer/schedule pairs without PyTorch."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

from experiments.synthetic_trainers.architecture_report import paired_normalizer_outcome, verify_archive
from experiments.synthetic_trainers.attention_layout import digest
from experiments.synthetic_trainers.persistence import PersistenceConfig
from experiments.synthetic_trainers.stability_recovery import describe
from experiments.synthetic_trainers.stability_recovery_series import timeline


def verify(directory):
    directory = Path(directory)
    expected = {str(p.relative_to(directory)): digest(p) for p in sorted(directory.rglob("*"))
                if p.is_file() and p != directory / "artifact-hashes.json"}
    if json.loads((directory / "artifact-hashes.json").read_text())["files"] != expected:
        raise ValueError("Architecture CPU preparation checksums differ")
    plan = json.loads((directory / "plan.json").read_text())
    validation = json.loads((directory / "validation.json").read_text())
    if (plan["scientific_run"] or not plan["CPU_fixture"] or plan["campaign_architecture_claim_allowed"]
            or plan["planned_runs"] != 4 or validation["scientific_run"] or not validation["CPU_fixture"]
            or validation["scientific_architecture_selected"] or plan["maximum_updates_per_run"] != 300000):
        raise ValueError("Architecture CPU fixtures cannot substitute for a scientific benchmark/campaign")
    names = [f"{schedule}-{norm}" for schedule in ("constant", "cosine_tail") for norm in ("sparsemax", "softmax")]
    if [row["name"] for row in plan["recipes"]] != names or plan["execution_order"] != names:
        raise ValueError("Architecture CPU preparation lacks both complete normalizer pairs")
    criterion = PersistenceConfig(**plan["criterion"])
    if asdict(criterion) != plan["criterion"]:
        raise ValueError("Architecture CPU criterion changed")
    summaries, recovery, cases = {}, {}, []
    for name in names:
        output = directory / name
        if json.loads((output / "plan.json").read_text()) != plan:
            raise ValueError("A normalizer case differs from the complete prospective CPU plan")
        summary = summaries[name] = verify_archive(output)
        config = summary["config"]
        if (config["device"] != "cpu" or config["steps"] != 40 or config["width"] != 128
                or config["layers"] != 2 or config["heads"] != 4 or summary["parameters"] != 412296):
            raise ValueError("Architecture CPU execution differs from its documented full-width fixture")
        cases.append({"name": name, "parameters": summary["parameters"], "completed_updates": config["steps"],
                      "stable_grokking": summary["assessment"]["stable_grokking"],
                      "canonical_observations": summary["canonical_observations"],
                      "gradient_observations": summary["gradient_trace"]["observations"]})
        measurements = json.loads((output / "measurements.json").read_text())
        recovery[name] = {"tail_and_episodes": describe(measurements, criterion, window_steps=10),
                          "whole_post_onset_series": timeline(measurements, criterion, window_steps=10)}
    pairs = {schedule: paired_normalizer_outcome(summaries[f"{schedule}-softmax"], summaries[f"{schedule}-sparsemax"])
             for schedule in ("constant", "cosine_tail")}
    if pairs != json.loads((directory / "architecture-pairs.json").read_text()):
        raise ValueError("Architecture pair outcomes differ from all complete native histories")
    if recovery != json.loads((directory / "recovery.json").read_text()):
        raise ValueError("Architecture recovery metrics differ from all complete histories")
    if cases != validation["cases"] or validation["total_completed_updates"] != sum(row["completed_updates"] for row in cases):
        raise ValueError("Architecture CPU validation differs from all complete cases")
    if set(validation["scientific_benchmark_guard_rejections"]) != {"normalizer", "schedule"}:
        raise ValueError("The CPU preparation lacks its recorded scientific benchmark refusals")
    if "torch" in sys.modules:
        raise RuntimeError("Architecture portable verification imported PyTorch")
    return {"CPU_cases": 4, "CPU_pairs": 2, "completed_updates": validation["total_completed_updates"],
            "all_stable_grokking": all(row["stable_grokking"] for row in cases),
            "eligible_pair_support": {name: row["eligible_pair_support"] for name, row in pairs.items()},
            "Torch_imported": False, "scientific_run": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.directory)))


if __name__ == "__main__":
    main()
