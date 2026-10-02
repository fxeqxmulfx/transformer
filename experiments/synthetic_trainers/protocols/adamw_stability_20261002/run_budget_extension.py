"""Continue the frozen native AdamW checkpoint to a total of 300,000 updates.

Convexifying Transformers, Section 4: preserve failed original observations;
this exploratory extension needs independent fresh confirmations after passing.
"""

import argparse
import fcntl
import json
from pathlib import Path

import torch

from experiments.synthetic_trainers.paper_reproduction.diagnostics import DiagnosticsConfig
from experiments.synthetic_trainers.paper_reproduction.grokking import RunConfig, train
from experiments.synthetic_trainers.persistence import PersistenceConfig
from experiments.synthetic_trainers.runtime import write_json
from experiments.synthetic_trainers.stability import freeze, validate_complete
from experiments.synthetic_trainers.stability_recovery import describe
from experiments.synthetic_trainers.stability_recovery_series import timeline
from .budget_extension_guard import read_frozen, verify_child, verify_prefixes


def checked_manifest():
    stage, expected = read_frozen()
    manifest = freeze(stage, expected["recipes"], DiagnosticsConfig(**expected["instrumentation"]),
                      PersistenceConfig(**expected["criterion"]), resume=True)
    run = verify_child(stage, manifest)
    return stage, manifest, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    stage, manifest, run = checked_manifest()
    if args.check_only:
        print("Frozen total 300,000-update cap, exact parent/native checkpoint, fixed windows and sources match.")
        return
    state = {"status": "running", "planned_runs": 1, "completed_runs": 0,
             "current_run": manifest["recipes"][0]["name"], "runs": []}
    peaks_path = stage / "observed-extension-peaks.json"
    peaks = json.loads(peaks_path.read_text()) if peaks_path.exists() else {
        key: 0 for key in ("peak_cuda_allocated_bytes", "peak_cuda_reserved_bytes")}
    original = json.loads((Path(manifest["parent_source_directory"]) / "measurements.json").read_text())

    def progress(row):
        if not row.get("diagnostic_probe"):
            for key, measure in (("peak_cuda_allocated_bytes", torch.cuda.max_memory_allocated),
                                 ("peak_cuda_reserved_bytes", torch.cuda.max_memory_reserved)):
                peaks[key] = max(peaks[key], measure(manifest["recipes"][0]["config"]["device"]))
            write_json(peaks_path, peaks)
        print(json.dumps(row), flush=True)

    with (stage / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another process is executing this budget extension") from error
        write_json(stage / "state.json", state)
        try:
            report_path = run / "measurements.json"
            report = json.loads(report_path.read_text()) if report_path.exists() else train(
                RunConfig(**manifest["recipes"][0]["config"]), run, resume=True,
                diagnostics=DiagnosticsConfig(**manifest["instrumentation"]), progress=progress)
            verify_prefixes(stage, manifest)
            provenance = {"parent_checkpoint_sha256": manifest["parent_checkpoint_sha256"],
                "parent_completed_steps": 150000, "total_frozen_steps": 300000,
                "original_log_prefixes_byte_preserved": True,
                "cost_scope": "cumulative_checkpoint_training_diagnostic_and_execution_wall; intervening_downtime_excluded",
                "memory_scope": "maximum_of_original_parent_and_all_observed_extension_segments",
                "extension_observed_peaks": peaks,
                "parent_peaks": {key: original[key] for key in peaks},
                "scope": "exploratory_posthoc_budget_extension; additional_updates_prospectively_frozen"}
            for key in peaks:
                report[key] = max(report[key], original[key], peaks[key])
            report["budget_extension_provenance"] = provenance
            write_json(report_path, report)
            assessment = validate_complete(report, manifest["recipes"][0], manifest)
            criterion = PersistenceConfig(**manifest["criterion"])
            write_json(stage / "recovery-metrics.json", describe(report, criterion, window_steps=manifest["recovery_window_steps"]))
            write_json(stage / "recovery-series.json", timeline(report, criterion, window_steps=manifest["recovery_window_steps"]))
            result = {"name": state["current_run"], "result": f'{state["current_run"]}/measurements.json',
                      "assessment": assessment, "training_seconds": report["training_seconds"],
                      "wall_seconds": report["wall_seconds"], "final": report["final"]}
        except BaseException as error:
            write_json(stage / "state.json", {**state, "status": "failed", "error": repr(error)})
            raise
        write_json(stage / "state.json", {**state, "status": "complete", "completed_runs": 1,
                                          "current_run": None, "runs": [result]})
        print(json.dumps({"event": "completed_run", **result}), flush=True)


if __name__ == "__main__":
    main()
