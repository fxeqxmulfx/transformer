"""Archive all 300,000 frozen updates and additional metrics without PyTorch."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments.synthetic_trainers.persistence import PersistenceConfig
from experiments.synthetic_trainers.stability_comparison import assemble
from experiments.synthetic_trainers.stability_recovery import describe
from experiments.synthetic_trainers.stability_recovery_series import timeline
from experiments.synthetic_trainers.stability_report import save_run
from .budget_extension_guard import digest, read_frozen, verify_prefixes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trainer-pid", required=True, type=int)
    args = parser.parse_args()
    stage, manifest = read_frozen()
    protocol = Path(__file__).resolve().parent
    name = manifest["recipes"][0]["name"]
    source = stage / name / "measurements.json"
    metadata = stage.parent / "bootstrap/budget300k-archive-worker.json"
    root = Path("experiments/synthetic_trainers/baselines")
    archive = root / "adamw_stability_mod193_fraction25_lr0003_budget300k_seed0_data0_20261002"
    comparison = root / "adamw_stability_mod193_fraction25_budget300k_calibration_20261002"
    additional = protocol / "budget300k-additional-metrics"

    def record(status, **values):
        value = {"recorded_at_utc": datetime.now(timezone.utc).isoformat(), "pid": os.getpid(),
                 "trainer_pid": args.trainer_pid, "status": status, "torch_imported": "torch" in sys.modules,
                 "worker_sha256": digest(Path(__file__)), **values}
        temporary = metadata.with_suffix(".tmp")
        temporary.write_text(json.dumps(value, indent=2) + "\n")
        temporary.replace(metadata)
        print(json.dumps(value), flush=True)

    def offline(path, combined=False):
        module, function = ("stability_comparison", "verify_comparison") if combined else ("stability_report", "verify_archive")
        code = (f"from experiments.synthetic_trainers.{module} import {function}; import sys,json; "
                + function + "(sys.argv[1]); print(json.dumps({'verified':True,'torch_imported':'torch' in sys.modules}))")
        value = json.loads(subprocess.check_output(["/usr/bin/python3", "-c", code, str(path)], text=True))
        if not value["verified"] or value["torch_imported"]:
            raise ValueError("The complete budget must verify without PyTorch")

    try:
        if "torch" in sys.modules:
            raise RuntimeError("Archive worker must not import PyTorch")
        command = Path(f"/proc/{args.trainer_pid}/cmdline").read_bytes().split(b"\x00")
        if b"experiments.synthetic_trainers.protocols.adamw_stability_20261002.run_budget_extension" not in command:
            raise ValueError("The supplied process is not the frozen budget-extension trainer")
        record("waiting_for_complete_300000_update_budget", scientific_training_started=True)
        while True:
            state_path = stage / "state.json"
            state = json.loads(state_path.read_text()) if state_path.exists() else {"status": "not_started"}
            if state["status"] == "complete" and source.exists():
                break
            if state["status"] == "failed":
                raise RuntimeError("Budget extension failed; preserve the incomplete history")
            if not Path(f"/proc/{args.trainer_pid}/cmdline").exists():
                raise RuntimeError("Trainer disappeared before the whole stage completed")
            time.sleep(40)
        read_frozen()
        verify_prefixes(stage, manifest)
        report = json.loads(source.read_text())
        if report["completed_steps"] != 300000 or report["plan"]["status"] != "complete":
            raise ValueError("Extended budget is incomplete")
        criterion = PersistenceConfig(**manifest["criterion"])
        for filename, expected in (("recovery-metrics.json", describe(report, criterion, window_steps=10000)),
                                   ("recovery-series.json", timeline(report, criterion, window_steps=10000))):
            if json.loads((stage / filename).read_text()) != expected:
                raise ValueError("Additional recovery metrics differ from the complete history")
        summary = save_run(stage, name, archive)
        offline(archive)
        assemble([archive], comparison)
        offline(comparison, combined=True)
        if additional.exists():
            raise FileExistsError("Additional complete metrics require a fresh destination")
        additional.mkdir()
        for filename in ("recovery-metrics.json", "recovery-series.json", "observed-extension-peaks.json"):
            (additional / filename).write_bytes((stage / filename).read_bytes())
        provenance = {"archive": str(archive), "measurement_sha256": digest(source),
                      "recovery_source_hashes": manifest["recovery_source_hashes"], "window_steps": 10000,
                      "budget_extension_provenance": report["budget_extension_provenance"]}
        (additional / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
        hashes = {p.name: digest(p) for p in additional.iterdir()}
        (additional / "artifact-hashes.json").write_text(json.dumps({"files": hashes}, indent=2) + "\n")
        read_frozen()
        record("budget300k_archives_ready_for_visual_review_and_commit", completed_updates=300000,
               archive=str(archive), comparison=str(comparison), offline_verification_passed=True,
               stable_grokking=summary["assessment"]["stable_grokking"],
               tail_failures=len(summary["assessment"]["tail_failures"]),
               gradient_trace_observations=summary["gradient_trace"]["observations"])
    except BaseException as error:
        record("failed", error=repr(error))
        raise


if __name__ == "__main__":
    main()
