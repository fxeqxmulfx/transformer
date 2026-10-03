"""Verify the complete scientific normalizer pair without Torch or updates."""

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from experiments.synthetic_trainers.attention_layout import digest, validate_pair_plan
from experiments.synthetic_trainers.attention_report import verify_pair
from experiments.synthetic_trainers.stability_comparison import hashes


PROTOCOL = Path(__file__).resolve().parent
ARCHIVES = Path("experiments/synthetic_trainers/baselines")
PAIR = ARCHIVES / "adamw_attention_softmax_sparsemax_mod193_fraction25_lr0003_budget300k_pair_20261002"


def main():
    plan = validate_pair_plan(json.loads((PROTOCOL / "attention-pair-plan.json").read_text()))
    raw = Path(plan["output_directory"])
    assert json.loads((raw / "plan.json").read_text()) == plan
    state = json.loads((raw / "state.json").read_text())
    assert state["status"] == "complete" and len(state["runs"]) == 2
    result = verify_pair(PAIR)
    assert result["complete_pair"] and result["scientific_run"]
    assert result["outcome"]["eligible_pair_support"] == 0
    assert not result["outcome"]["control"]["stable_grokking"]
    assert not result["campaign_architecture_claim_allowed"]
    counts = {}
    for recipe in plan["recipes"]:
        name = recipe["name"]
        case = raw / name
        report = json.loads((case / "measurements.json").read_text())
        assert report["completed_steps"] == 300000 and len(report["history"]) == 1201
        archive = ARCHIVES / f"adamw_attention_{name}_mod193_fraction25_lr0003_budget300k_seed0_data0_20261002"
        assert hashes(archive) == hashes(PAIR / "runs" / name)
        row = {"canonical": 1201}
        for filename, expected in (("probes.jsonl", 2400), ("diagnostics.jsonl", 3600), ("gradients.jsonl", 300000)):
            assert digest(case / filename) == digest(archive / filename)
            row[filename] = sum(1 for _ in (case / filename).open())
            assert row[filename] == expected
        assert digest(case / "measurements.json") == digest(archive / "measurements.json")
        counts[name] = row
    for field in ("source_hashes", "training_source_hashes", "Lean_specification_hashes", "papers"):
        assert all(digest(path) == expected for path, expected in plan[field].items())
    assert digest(PROTOCOL / "sparsemax-preparation/validation.json") == plan["Lean_audit_validation_sha256"]
    assert hashes(PAIR / "reference") == plan["reference_hashes"]
    reference = json.loads((PAIR / "reference/runs/adamw-mod193-fraction25-lr0003-budget300k/measurements.json").read_text())
    softmax = json.loads((raw / "adamw-softmax/measurements.json").read_text())
    time_keys = {"training_seconds", "wall_seconds"}
    strip = lambda rows: [{k: v for k, v in row.items() if k not in time_keys} for row in rows]
    assert strip(softmax["history"]) == strip(reference["history"])
    assert "torch" not in sys.modules
    receipt = {"recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "complete_two_run_exploratory_normalizer_pair; independent_benchmark_gate_closed",
        "completed_updates_per_run": 300000, "completed_scientific_runs": 2,
        "offline_verification_passed": True, "Torch_imported": False,
        "all_raw_histories_and_case_archives_preserved_byte_for_byte": True,
        "reference_archive_preserved_byte_for_byte": True,
        "softmax_all_1201_non_time_canonical_observations_equal_frozen_same_seed_reference": True,
        "independent_confirmation_claimed": False, "observation_counts": counts,
        "all_43_Python_17_training_9_Lean_2_papers_and_audit_unchanged": True,
        "archive_artifact_manifest_sha256": digest(PAIR / "artifact-hashes.json"),
        "verification_script_sha256": digest(__file__), "outcome": result["outcome"],
        "PNG_figures_visually_reviewed": False, "PDF_figures_visually_reviewed": False,
        "trainer_and_archive_worker_terminal_sessions_consumed": False,
        "native_optimizer_CPU_audits": None, "campaign_architecture_claim_allowed": False}
    output = PROTOCOL / "attention-pair-result-validation.json"
    assert not output.exists()
    output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"complete_pair": True, "full_updates": 600000,
                      "eligible_pair_support": 0, "offline_verification_passed": True}))


if __name__ == "__main__":
    main()
