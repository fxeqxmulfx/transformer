"""Wait for the complete parent pair, run the frozen harder task, archive it."""
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
from experiments.synthetic_trainers.stability_comparison import assemble, verify_comparison
from experiments.synthetic_trainers.stability_report import save_run

protocol = repository / "experiments/synthetic_trainers/protocols/adamw_stability_20261002"
plan = json.loads((protocol / "larger-modulus-plan.json").read_text())
parent = json.loads((protocol / "optimizer-pair-plan.json").read_text())
stage = repository / plan["output_directory"]
parent_stage = repository / parent["output_directory"]
parent_archive = repository / plan["parent_complete_comparison"]
metadata = stage.parent / "bootstrap/queued-mod193-worker.json"
driver_log = stage.parent / "mod193-driver.log"
archive = repository / "experiments/synthetic_trainers/baselines/adamw_stability_mod193_lr001_seed0_data0_20261002"
comparison = repository / "experiments/synthetic_trainers/baselines/adamw_stability_mod193_calibration_20261002"


def record(status, **values):
    value = {"recorded_at_utc": datetime.now(timezone.utc).isoformat(), "pid": os.getpid(),
             "status": status, "torch_imported": "torch" in sys.modules,
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
        raise ValueError("Larger-modulus manifest changed")
    if hashlib.sha256((protocol / "optimizer-pair-plan.json").read_bytes()).hexdigest() != plan["parent_optimizer_pair_plan_sha256"]:
        raise ValueError("Parent manifest changed")
    for values in (plan["source_hashes"], plan["analysis_and_driver_source_hashes"], plan["papers"]):
        if any(hashlib.sha256((repository / path).read_bytes()).hexdigest() != value for path, value in values.items()):
            raise ValueError("Larger-modulus frozen sources or manuscripts changed")
    if hashlib.sha256((protocol / "run_larger_modulus.py").read_bytes()).hexdigest() != plan["launcher_sha256"]:
        raise ValueError("Larger-modulus launcher changed")


def offline(path, combined=False):
    result = subprocess.run(["/usr/bin/python3", "-m", csv_adapter.__name__,
                   "comparison" if combined else "archive", str(path)], cwd=repository,
                   check=True, capture_output=True, text=True)
    verified = json.loads(result.stdout)
    if not verified["verified"] or verified["torch_imported"]:
        raise ValueError("Portable verification did not complete without PyTorch")

with linear_csv_verification():
    try:
        fingerprints()
        record("queued_after_complete_verified_mod97_optimizer_pair", parent_trainer_pid=112572,
               parent_archive=str(parent_archive.relative_to(repository)), scientific_training_started=False)
        while not parent_archive.exists():
            state = json.loads((parent_stage / "state.json").read_text())
            worker = json.loads((stage.parent / "bootstrap/optimizer-pair-archive-worker.json").read_text())
            if state["status"] == "failed" or worker["status"] == "failed":
                raise RuntimeError("Parent training or archive worker failed; harder task remains unstarted")
            if state["status"] == "running" and not Path("/proc/112572/cmdline").exists():
                raise RuntimeError("Parent trainer disappeared before its budgets completed")
            time.sleep(40)
        verify_comparison(parent_archive)
        if json.loads((parent_archive / "plan.json").read_text()) != parent:
            raise ValueError("Complete parent comparison differs from its frozen plan")
        fingerprints()
        with driver_log.open("a") as log:
            job = subprocess.Popen([str(repository / ".venv/bin/python"), "-u", "-m",
                "experiments.synthetic_trainers.protocols.adamw_stability_20261002.csv_verification", "module",
                "experiments.synthetic_trainers.protocols.adamw_stability_20261002.run_larger_modulus"],
                cwd=repository, stdout=log, stderr=subprocess.STDOUT)
            record("mod193_scientific_training_running", trainer_pid=job.pid, scientific_training_started=True)
            while job.poll() is None:
                time.sleep(40)
            if job.returncode != 0:
                raise RuntimeError(f"Larger-modulus trainer exited with code {job.returncode}")
        fingerprints()
        summary = save_run(stage, "adamw-mod193-lr001", archive)
        offline(archive)
        result = assemble([archive], comparison)
        offline(comparison, combined=True)
        fingerprints()
        record("mod193_archives_ready_for_visual_review_and_commit", scientific_training_started=True,
               completed_updates=150000, archive=str(archive.relative_to(repository)),
               comparison=str(comparison.relative_to(repository)), offline_verification_passed=True,
               stable_grokking=summary["assessment"]["stable_grokking"],
               tail_failures=len(summary["assessment"]["tail_failures"]),
               gradient_trace_observations=summary["gradient_trace"]["observations"])
    except BaseException as error:
        record("failed", error=repr(error))
        raise
