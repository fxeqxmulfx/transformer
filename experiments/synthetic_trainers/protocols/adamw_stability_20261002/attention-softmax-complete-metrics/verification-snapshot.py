"""Verify a full scientific normalizer archive against its native raw case.

Run with system Python: no Torch, checkpoint load or CUDA context is needed.
The emitted receipt starts with visual/checkpoint audit flags unset; review
the actual figures and native state before committing a completed result.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

from experiments.synthetic_trainers.attention_layout import digest, validate_pair_plan
from experiments.synthetic_trainers.persistence import PersistenceConfig
from experiments.synthetic_trainers.stability_recovery import describe
from experiments.synthetic_trainers.stability_recovery_series import timeline
from experiments.synthetic_trainers.stability_report import verify_archive, write_json


PROTOCOL = Path(__file__).resolve().parent


def verify(name):
    plan = validate_pair_plan(json.loads((PROTOCOL / "attention-pair-plan.json").read_text()))
    raw = Path(plan["output_directory"])
    case = raw / name
    archive = Path("experiments/synthetic_trainers/baselines") / (
        f"adamw_attention_{name}_mod193_fraction25_lr0003_budget300k_seed0_data0_20261002")
    output = PROTOCOL / f"attention-{name.removeprefix('adamw-')}-complete-metrics"
    if output.exists():
        raise FileExistsError("A complete result receipt requires a fresh destination")
    assert json.loads((raw / "plan.json").read_text()) == plan
    state = json.loads((raw / "state.json").read_text())
    assert name in [row["name"] for row in state["runs"]]
    summary = verify_archive(archive)
    report = json.loads((case / "measurements.json").read_text())
    recipe = next(row for row in plan["recipes"] if row["name"] == name)
    assert summary["config"] == report["plan"]["config"] == recipe["config"]
    assert report["completed_steps"] == recipe["config"]["steps"] == 300000
    assert summary["parameters"] == report["plan"]["parameters"] == 436104
    assert report["plan"]["source_hashes"] == plan["training_source_hashes"]
    counts = {"canonical": len(report["history"])}
    assert counts["canonical"] == 1201
    for filename, expected in (("probes.jsonl", 2400), ("diagnostics.jsonl", 3600), ("gradients.jsonl", 300000)):
        assert (case / filename).read_bytes() == (archive / filename).read_bytes()
        count = sum(1 for _ in (archive / filename).open())
        assert count == expected
        counts[filename] = count
    assert (case / "measurements.json").read_bytes() == (archive / "measurements.json").read_bytes()
    assert [json.loads(line) for line in (case / "history.jsonl").read_text().splitlines()] == report["history"]
    for field in ("source_hashes", "training_source_hashes", "Lean_specification_hashes", "papers"):
        assert all(digest(filename) == expected for filename, expected in plan[field].items())
    assert digest(PROTOCOL / "sparsemax-preparation/validation.json") == plan["Lean_audit_validation_sha256"]
    observed = json.loads((case / "observed-peaks.json").read_text())
    assert all(report[key] >= value for key, value in observed.items())
    assert summary["assessment"]["complete_canonical_history"]
    assert summary["assessment"]["tail_observations"] == 201
    criterion = PersistenceConfig(**plan["criterion"])
    recovery, series = describe(report, criterion, window_steps=10000), timeline(report, criterion, window_steps=10000)
    assert json.loads((case / "recovery-metrics.json").read_text()) == recovery
    assert json.loads((case / "recovery-series.json").read_text()) == series
    assert "torch" not in sys.modules
    output.mkdir()
    shutil.copyfile(__file__, output / "verification-snapshot.py")
    for filename in ("recovery-metrics.json", "recovery-series.json"):
        shutil.copyfile(case / filename, output / filename)
    receipt = {"recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "one_complete_exploratory_normalizer_case; paired_control_still_required",
        "name": name, "archive": str(archive), "completed_updates": 300000,
        "offline_verification_passed": True, "Torch_imported": False,
        "full_raw_logs_and_measurement_bytes_match_archive": True, "observation_counts": counts,
        "all_43_Python_17_training_9_Lean_2_papers_and_audit_unchanged": True,
        "parameters": 436104, "checkpoint_sha256": digest(case / "checkpoint.pt"),
        "archive_artifact_manifest_sha256": digest(archive / "artifact-hashes.json"),
        "peak_memory_includes_all_recorded_canonical_segment_peaks": True,
        "observed_peaks": observed, "assessment": summary["assessment"],
        "gradient_trace": summary["gradient_trace"],
        "final": report["final"], "training_seconds": report["training_seconds"],
        "wall_seconds": report["wall_seconds"], "diagnostic_seconds": report["diagnostic_seconds"],
        "native_optimizer_CPU_audit": None,
        "PNG_figures_visually_reviewed": False, "PDF_figures_visually_reviewed": False,
        "full_pair_complete": False, "independent_confirmation_gate_opened_by_this_case": False}
    write_json(output / "validation.json", receipt)
    write_json(output / "artifact-hashes.json", {"files": {str(p.relative_to(output)): digest(p)
        for p in sorted(output.rglob("*")) if p.is_file() and p.name != "artifact-hashes.json"}})
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", choices=("adamw-sparsemax", "adamw-softmax"))
    result = verify(parser.parse_args().name)
    print(json.dumps({"case": result["name"], "completed_updates": result["completed_updates"],
                      "offline_verification_passed": True, "final_heldout": result["final"]["heldout"]["accuracy"]}))
