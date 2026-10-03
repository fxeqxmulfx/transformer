"""Verify both complete CPU preparation cohorts without importing Torch."""

import argparse
import json
from pathlib import Path
import sys

from experiments.synthetic_trainers.architecture_cohort_report import verify_cohort
from experiments.synthetic_trainers.attention_layout import digest
from experiments.synthetic_trainers.stability_comparison import hashes


def verify(directory):
    directory = Path(directory)
    files = {str(p.relative_to(directory)): digest(p) for p in sorted(directory.rglob("*"))
             if p.is_file() and p != directory / "artifact-hashes.json"}
    assert files == json.loads((directory / "artifact-hashes.json").read_text())["files"]
    audits = json.loads((directory / "native-checkpoint-audits.json").read_text())
    outcomes = {}
    for mode, steps in (("normalizer", 20), ("schedule", 40)):
        root = directory / mode
        result = verify_cohort(root); plan = json.loads((root / "plan.json").read_text())
        assert result["completed_runs"] == 12 and result["eligible_pair_support"] == 0
        assert not result["scientific_run"] and not result["repeatable_architecture_improvement"]
        assert hashes(root / "benchmark") == plan["benchmark_hashes"]
        assert set(audits[mode]) == {r["name"] for r in plan["recipes"]}
        for recipe in plan["recipes"]:
            audit = audits[mode][recipe["name"]]
            artifact = json.loads((root / "runs" / recipe["name"] / "artifact-hashes.json").read_text())
            assert audit["checkpoint_sha256"] == artifact["raw_checkpoint_sha256"]
            assert audit["completed_updates"] == steps and audit["native_states"] == 11
            assert audit["native_model_and_moments_finite"] and audit["audit_optimizer_updates_performed"] == 0
        outcomes[mode] = result
    receipt = json.loads((directory / "validation.json").read_text())
    assert receipt["total_CPU_updates"] == sum(r["total_updates"] for r in outcomes.values()) == 720
    assert "torch" not in sys.modules
    return {"verified": True, "complete_CPU_cohorts": 2, "complete_CPU_pairs": 12,
            "complete_CPU_runs": 24, "total_CPU_updates": 720, "Torch_imported": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    print(json.dumps(verify(parser.parse_args().directory)))
