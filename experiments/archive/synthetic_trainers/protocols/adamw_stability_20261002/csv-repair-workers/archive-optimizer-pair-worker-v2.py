"""Archive each finished optimizer control, preserving frozen evidence."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

repository = Path.cwd()
sys.path.insert(0, str(repository))
import experiments.synthetic_trainers.protocols.adamw_stability_20261002.csv_verification as csv_adapter
from experiments.synthetic_trainers.protocols.adamw_stability_20261002.csv_verification import linear_csv_verification
csv_adapter_sha256 = hashlib.sha256(Path(csv_adapter.__file__).read_bytes()).hexdigest()
from experiments.synthetic_trainers.stability_report import save_run, verify_archive
from experiments.synthetic_trainers.stability_comparison import assemble, verify_comparison

protocol = repository / "experiments/synthetic_trainers/protocols/adamw_stability_20261002"
plan = json.loads((protocol / "optimizer-pair-plan.json").read_text())
stage = repository / plan["output_directory"]
metadata = stage.parent / "bootstrap/optimizer-pair-archive-worker.json"
archives = {
    "adamw-short-lr001": repository / "experiments/synthetic_trainers/baselines/adamw_stability_adamw_short_lr001_seed0_data0_20261002",
    "amsgradw-short-lr001": repository / "experiments/synthetic_trainers/baselines/adamw_stability_raw_amsgradw_short_lr001_seed0_data0_20261002",
}
comparison = repository / "experiments/synthetic_trainers/baselines/adamw_stability_optimizer_pair_lr001_20261002"
completed = []
os.nice(19)

def record(status, **values):
    value = {"recorded_at_utc": datetime.now(timezone.utc).isoformat(), "pid": os.getpid(),
             "status": status, "completed_archives": completed, "torch_imported": "torch" in sys.modules,
             "worker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             "csv_runtime_adapter_sha256": csv_adapter_sha256,
             "csv_execution": "unchanged_table_comparison_with_columns_computed_once", **values}
    temporary = metadata.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(metadata)
    print(json.dumps(value), flush=True)

def fingerprints():
    if hashlib.sha256(Path(csv_adapter.__file__).read_bytes()).hexdigest() != csv_adapter_sha256:
        raise ValueError("CSV runtime adapter changed during this worker")
    if json.loads((stage / "plan.json").read_text()) != plan:
        raise ValueError("Live optimizer plan changed")
    for values in (plan["source_hashes"], plan["analysis_and_driver_source_hashes"], plan["papers"]):
        if any(hashlib.sha256((repository / path).read_bytes()).hexdigest() != value for path, value in values.items()):
            raise ValueError("Frozen source, analysis or manuscript changed")
    if hashlib.sha256((protocol / "run_optimizer_pair.py").read_bytes()).hexdigest() != plan["launcher_sha256"]:
        raise ValueError("Frozen launcher changed")

def offline(path, combined=False):
    result = subprocess.run(["/usr/bin/python3", "-m", csv_adapter.__name__,
                   "comparison" if combined else "archive", str(path)], cwd=repository,
                   check=True, capture_output=True, text=True)
    verified = json.loads(result.stdout)
    if not verified["verified"] or verified["torch_imported"]:
        raise ValueError("Portable verification did not complete without PyTorch")

with linear_csv_verification():
    try:
        record("waiting_for_complete_AdamW_budget")
        for recipe in plan["recipes"]:
            report = stage / recipe["name"] / "measurements.json"
            while not report.exists():
                time.sleep(40)
                state = json.loads((stage / "state.json").read_text())
                if state["status"] == "failed":
                    raise RuntimeError("Optimizer trainer failed; preserve incomplete history")
            measurements = json.loads(report.read_text())
            if measurements["completed_steps"] != recipe["config"]["steps"] or measurements["plan"]["status"] != "complete":
                raise ValueError("Optimizer budget is not complete")
            fingerprints()
            archive = archives[recipe["name"]]
            summary = verify_archive(archive) if archive.exists() else save_run(stage, recipe["name"], archive)
            if json.loads((archive / "plan.json").read_text()) != plan:
                raise ValueError("Optimizer archive differs from the frozen plan")
            offline(archive)
            fingerprints()
            completed.append(str(archive.relative_to(repository)))
            record("individual_archive_ready_for_visual_review_and_commit", recipe=recipe["name"],
                   stable_grokking=summary["assessment"]["stable_grokking"],
                   tail_failures=len(summary["assessment"]["tail_failures"]),
                   gradient_trace_observations=summary["gradient_trace"]["observations"],
                   offline_verification_passed=True)
        summary = verify_comparison(comparison) if comparison.exists() else assemble(list(archives.values()), comparison)
        offline(comparison, combined=True)
        fingerprints()
        record("all_archives_ready_for_visual_review_and_commit", comparison=str(comparison.relative_to(repository)),
               offline_verification_passed=True, completed_runs=summary["completed_runs"],
               stable_grokking_calibration_recipes=summary["stable_grokking_calibration_recipes"])
    except BaseException as error:
        record("failed", error=repr(error))
        raise
