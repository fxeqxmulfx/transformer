"""Archive the full frozen fraction calibration without importing PyTorch."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments.synthetic_trainers.stability_comparison import assemble
from experiments.synthetic_trainers.stability_report import save_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trainer-pid", type=int, required=True)
    args = parser.parse_args()
    repository = Path.cwd()
    protocol = Path(__file__).resolve().parent
    plan = json.loads((protocol / "fraction25-plan.json").read_text())
    stage = repository / plan["output_directory"]
    source = stage / "adamw-mod193-fraction25-lr001/measurements.json"
    metadata = stage.parent / "bootstrap/fraction25-archive-worker.json"
    root = repository / "experiments/synthetic_trainers/baselines"
    archive = root / "adamw_stability_mod193_fraction25_lr001_seed0_data0_20261002"
    comparison = root / "adamw_stability_mod193_fraction25_calibration_20261002"

    def record(status, **values):
        value = {"recorded_at_utc": datetime.now(timezone.utc).isoformat(), "pid": os.getpid(),
                 "trainer_pid": args.trainer_pid, "status": status, "torch_imported": "torch" in sys.modules,
                 "worker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), **values}
        temporary = metadata.with_suffix(".tmp")
        temporary.write_text(json.dumps(value, indent=2) + "\n")
        temporary.replace(metadata)
        print(json.dumps(value), flush=True)

    def fingerprints():
        if json.loads((stage / "plan.json").read_text()) != plan:
            raise ValueError("Live fraction plan changed")
        for values in (plan["source_hashes"], plan["analysis_and_driver_source_hashes"], plan["papers"]):
            for name, value in values.items():
                if hashlib.sha256((repository / name).read_bytes()).hexdigest() != value:
                    raise ValueError("Frozen fraction source or manuscript changed")
        if hashlib.sha256((protocol / "run_fraction25.py").read_bytes()).hexdigest() != plan["launcher_sha256"]:
            raise ValueError("Fraction launcher changed")

    def offline(path, combined=False):
        module = "stability_comparison" if combined else "stability_report"
        function = "verify_comparison" if combined else "verify_archive"
        code = (f"from experiments.synthetic_trainers.{module} import {function}; "
                "import json,sys; " + function + "(sys.argv[1]); "
                "print(json.dumps({'verified':True,'torch_imported':'torch' in sys.modules}))")
        result = subprocess.run(["/usr/bin/python3", "-c", code, str(path)],
                                cwd=repository, check=True, capture_output=True, text=True)
        value = json.loads(result.stdout)
        if not value["verified"] or value["torch_imported"]:
            raise ValueError("Portable fraction verification must finish without PyTorch")

    try:
        fingerprints()
        command = Path(f"/proc/{args.trainer_pid}/cmdline").read_bytes().split(b"\x00")
        module = b"experiments.synthetic_trainers.protocols.adamw_stability_20261002.run_fraction25"
        if module not in command:
            raise ValueError("The supplied process is not the frozen fraction trainer")
        record("waiting_for_complete_fraction25_budget", scientific_training_started=True)
        while not source.exists():
            time.sleep(40)
            state = json.loads((stage / "state.json").read_text())
            if state["status"] == "failed":
                raise RuntimeError("Fraction training failed; preserve its incomplete history")
            if state["status"] == "running" and not Path(f"/proc/{args.trainer_pid}/cmdline").exists():
                raise RuntimeError("Fraction trainer disappeared before completing its budget")
        report = json.loads(source.read_text())
        if report["completed_steps"] != 150000 or report["plan"]["status"] != "complete":
            raise ValueError("The fraction budget is incomplete")
        fingerprints()
        summary = save_run(stage, "adamw-mod193-fraction25-lr001", archive)
        offline(archive)
        assemble([archive], comparison)
        offline(comparison, combined=True)
        fingerprints()
        record("fraction25_archives_ready_for_visual_review_and_commit", completed_updates=150000,
               archive=str(archive.relative_to(repository)), comparison=str(comparison.relative_to(repository)),
               offline_verification_passed=True, stable_grokking=summary["assessment"]["stable_grokking"],
               tail_failures=len(summary["assessment"]["tail_failures"]),
               gradient_trace_observations=summary["gradient_trace"]["observations"])
    except BaseException as error:
        record("failed", error=repr(error))
        raise


if __name__ == "__main__":
    main()
