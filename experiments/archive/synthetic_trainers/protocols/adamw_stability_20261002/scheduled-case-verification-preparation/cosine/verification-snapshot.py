"""Verify a completed fixed-schedule case and its archive without Torch."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

from experiments.synthetic_trainers.persistence import PersistenceConfig
from experiments.synthetic_trainers.scheduled_layout import current_sources, digest, validate_pair_plan
from experiments.synthetic_trainers.scheduled_report import verify_archive
from experiments.synthetic_trainers.stability_recovery import describe
from experiments.synthetic_trainers.stability_recovery_series import timeline
from experiments.synthetic_trainers.stability_report import write_json


PROTOCOL = Path("experiments/synthetic_trainers/protocols/adamw_stability_20261002")


def verify(stage, name, archive, output):
    """Preserve all raw bytes; native-state and visual checks remain separate."""
    if "torch" in sys.modules:
        raise RuntimeError("Portable case verification must not import Torch")
    stage, archive, output = Path(stage), Path(archive), Path(output)
    if output.exists():
        raise FileExistsError("A complete case receipt requires a fresh destination")
    plan = validate_pair_plan(json.loads((stage / "plan.json").read_text()))
    if plan["scientific_run"] and plan != json.loads((PROTOCOL / "scheduled-pair-plan.json").read_text()):
        raise ValueError("Scientific verification requires the exact committed prospective plan")
    state = json.loads((stage / "state.json").read_text())
    case = stage / name
    if (name not in [row["name"] for row in state["runs"]]
            or not (case / "measurements.json").exists()):
        raise ValueError("Portable verification requires the complete frozen case")
    recipe = next(row for row in plan["recipes"] if row["name"] == name)
    report = json.loads((case / "measurements.json").read_text())
    summary = verify_archive(archive)
    steps = recipe["config"]["steps"]
    if (report["completed_steps"] != steps
            or report["plan"]["config"] != recipe["config"] or summary["config"] != recipe["config"]
            or report["plan"]["source_hashes"] != plan["training_source_hashes"]
            or summary["parameters"] != report["plan"]["parameters"]):
        raise ValueError("Complete budget, configuration, parameters or native sources differ")
    if plan["scientific_run"] and (steps != 300000 or summary["parameters"] != 436104):
        raise ValueError("Scientific schedule verification retains the frozen full budget/model")
    if (case / "measurements.json").read_bytes() != (archive / "measurements.json").read_bytes():
        raise ValueError("Raw measurements differ from the complete archive")
    history = [json.loads(line) for line in (case / "history.jsonl").read_text().splitlines()]
    if history != report["history"]:
        raise ValueError("Raw canonical JSONL differs from the complete measurements")
    counts = {"canonical": len(history)}
    for filename in ("probes.jsonl", "diagnostics.jsonl", "gradients.jsonl"):
        if digest(case / filename) != digest(archive / filename):
            raise ValueError("Raw neighboring/tensor/gradient log bytes differ")
        counts[filename] = sum(1 for _ in (archive / filename).open())
    if counts["gradients.jsonl"] != steps:
        raise ValueError("Every actual update must retain its dense gradient/rate record")
    if plan["scientific_run"] and counts != {
            "canonical": 1201, "probes.jsonl": 2400, "diagnostics.jsonl": 3600, "gradients.jsonl": 300000}:
        raise ValueError("Scientific full-budget observation counts differ")
    if current_sources() != plan["source_hashes"]:
        raise ValueError("The frozen 55 training/protocol sources changed")
    for field in ("training_source_hashes", "Lean_specification_hashes", "papers"):
        if any(digest(path) != expected for path, expected in plan[field].items()):
            raise ValueError("Frozen native, Lean or manuscript fingerprints changed")
    if digest(PROTOCOL / "sparsemax-preparation/validation.json") != plan["Lean_audit_validation_sha256"]:
        raise ValueError("The frozen Lean audit receipt changed")
    criterion = PersistenceConfig(**plan["criterion"])
    recovery = describe(report, criterion, window_steps=plan["recovery_window_steps"])
    series = timeline(report, criterion, window_steps=plan["recovery_window_steps"])
    for filename, expected in (("recovery-metrics.json", recovery), ("recovery-series.json", series)):
        if json.loads((case / filename).read_text()) != expected:
            raise ValueError("Saved recovery derivation differs from the complete observations")
    observed = json.loads((case / "observed-peaks.json").read_text()) if (case / "observed-peaks.json").exists() else {}
    if plan["scientific_run"] and (not observed or any(value is None for value in observed.values())):
        raise ValueError("Scientific cases require actual recorded peak-memory observations")
    if any(value is not None and report[key] < value for key, value in observed.items()):
        raise ValueError("Reported peak memory omits an observed segment peak")
    manifest = json.loads((archive / "artifact-hashes.json").read_text())
    checkpoint_hash = digest(case / "checkpoint.pt")
    if manifest["raw_checkpoint_sha256"] != checkpoint_hash:
        raise ValueError("Actual raw checkpoint fingerprint differs from its complete archive")
    if not summary["assessment"]["complete_canonical_history"]:
        raise ValueError("Complete canonical evaluation grid is required")
    assert "torch" not in sys.modules
    output.mkdir(parents=True)
    shutil.copyfile(__file__, output / "verification-snapshot.py")
    for filename in ("recovery-metrics.json", "recovery-series.json"):
        shutil.copyfile(case / filename, output / filename)
    receipt = {"recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "complete_exploratory_schedule_case; independent_gates_remain_external"
                 if plan["scientific_run"] else "complete_CPU_case_verification_fixture; no_learning_result",
        "name": name, "scientific_run": plan["scientific_run"], "archive": str(archive),
        "completed_updates": steps, "parameters": summary["parameters"],
        "offline_verification_passed": True, "Torch_imported": False,
        "full_raw_logs_and_measurement_bytes_match_archive": True, "observation_counts": counts,
        "all_55_Python_17_training_9_Lean_2_papers_and_audit_unchanged": True,
        "checkpoint_sha256": checkpoint_hash,
        "archive_artifact_manifest_sha256": digest(archive / "artifact-hashes.json"),
        "observed_peaks": observed, "peak_memory_checked_against_recorded_segments": bool(observed),
        "assessment": summary["assessment"], "gradient_trace": summary["gradient_trace"],
        "final": report["final"], "training_seconds": report["training_seconds"],
        "wall_seconds": report["wall_seconds"], "diagnostic_seconds": report["diagnostic_seconds"],
        "native_optimizer_CPU_audit": None,
        "PNG_figures_visually_reviewed": False, "PDF_figures_visually_reviewed": False,
        "full_pair_complete": state["status"] == "complete",
        "independent_confirmation_gate_opened_by_this_case": False}
    write_json(output / "validation.json", receipt)
    write_json(output / "artifact-hashes.json", {"files": {str(path.relative_to(output)): digest(path)
        for path in sorted(output.rglob("*")) if path.is_file() and path.name != "artifact-hashes.json"}})
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=("adamw-constant", "adamw-cosine-tail"))
    parser.add_argument("--stage", type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    stage = args.stage or Path(json.loads((PROTOCOL / "scheduled-pair-plan.json").read_text())["output_directory"])
    archive = args.archive or Path("experiments/synthetic_trainers/baselines") / (
        f"adamw_schedule_{args.case}_mod193_fraction25_lr0003_budget300k_seed0_data0_20261002")
    output = args.output or PROTOCOL / f"scheduled-{args.case.removeprefix('adamw-')}-complete-metrics"
    result = verify(stage, args.case, archive, output)
    print(json.dumps({"case": args.case, "completed_updates": result["completed_updates"],
        "scientific_run": result["scientific_run"], "offline_verification_passed": True,
        "Torch_imported": False}))


if __name__ == "__main__":
    main()
