"""Freeze the user-requested total 300,000-update continuation before training."""

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import torch

from experiments.synthetic_trainers.confirmation_protocol import confirmation_sources
from experiments.synthetic_trainers.paper_reproduction.diagnostics import DiagnosticsConfig
from experiments.synthetic_trainers.persistence import PersistenceConfig
from experiments.synthetic_trainers.runtime import write_json
from experiments.synthetic_trainers.stability import freeze
from experiments.synthetic_trainers.stability_integrity import expected_batches
from .budget_extension_guard import digest


def main():
    protocol = Path(__file__).resolve().parent
    committed = protocol / "budget300k-plan.json"
    parent_path = protocol / "lower-rate-plan.json"
    parent = json.loads(parent_path.read_text())
    source = Path(parent["output_directory"]) / parent["recipes"][0]["name"]
    archive = Path("experiments/synthetic_trainers/baselines/adamw_stability_mod193_fraction25_lr0003_seed0_data0_20261002")
    comparison = Path("experiments/synthetic_trainers/baselines/adamw_stability_mod193_fraction25_lower_rate_calibration_20261002")
    stage = Path("experiments/runs/adamw_stability_20261002/calibration_mod193_fraction25_lr0003_budget300k")
    if committed.exists() or stage.exists():
        raise FileExistsError("Budget extension requires fresh plan and stage destinations")
    if confirmation_sources() != parent["analysis_and_driver_source_hashes"]:
        raise ValueError("Original training/analysis sources changed")
    preparation = json.loads((protocol / "budget-extension-preparation/validation.json").read_text())
    config = deepcopy(parent["recipes"][0]["config"])
    config["steps"] = 300000
    if preparation["scientific_configuration_after"] != config:
        raise ValueError("CPU continuation did not verify this exact scientific configuration")
    report = json.loads((source / "measurements.json").read_text())
    if report["completed_steps"] != 150000 or report["plan"]["status"] != "complete":
        raise ValueError("Original parent must finish before extending its budget")
    checkpoint = torch.load(source / "checkpoint.pt", map_location="cpu", weights_only=True)
    original_batches = expected_batches(parent["recipes"][0]["config"], 150000)
    if (checkpoint["step"] != 150000 or checkpoint["examples_seen"] != original_batches["examples_seen"]
            or checkpoint["cursor"] != original_batches["cursor_after"]):
        raise ValueError("Original checkpoint step/exposure/cursor do not match")
    if (not checkpoint["optimizer"]["state"]
            or any(float(s["step"]) != 150000 for s in checkpoint["optimizer"]["state"].values())):
        raise ValueError("Original native AdamW parameter steps are incomplete")
    if sorted(checkpoint["permutation"].tolist()) != list(range(parent["corpus"]["train_examples"])):
        raise ValueError("Original sampling permutation is incomplete")
    recipe = {"name": "adamw-mod193-fraction25-lr0003-budget300k", "config": config,
              "changed_mechanism": "total_update_budget_only; native_checkpoint_continuation"}
    manifest = freeze(stage, [recipe], DiagnosticsConfig(**parent["instrumentation"]),
                      PersistenceConfig(**parent["criterion"]))
    for key in ("source_hashes", "training_source_hashes", "instrumentation", "environment", "papers", "corpus"):
        if manifest[key] != parent[key]:
            raise ValueError(f"The continuation changed {key}")
    files = ("plan.json", "checkpoint.pt", "history.jsonl", "gradients.jsonl", "diagnostics.jsonl", "probes.jsonl", "measurements.json")
    original_hashes = {name: digest(source / name) for name in files}
    copied_hashes = {name: value for name, value in original_hashes.items() if name != "measurements.json"}
    run = stage / recipe["name"]
    run.mkdir()
    for name in copied_hashes:
        shutil.copyfile(source / name, run / name)
        if digest(run / name) != copied_hashes[name]:
            raise ValueError("Continuation copy differs from the original checkpoint/history")
    extension_names = ("budget_extension_guard.py", "run_budget_extension.py", "archive_budget_extension.py",
                       "prepare_budget_extension.py", "freeze_budget_extension.py")
    recovery_paths = (Path("experiments/synthetic_trainers/stability_recovery.py"),
                      Path("experiments/synthetic_trainers/stability_recovery_series.py"), protocol / "plot_recovery.py")
    manifest.update({"stage": "calibration_native_AdamW_budget_extension", "primary_optimizer": "adamw",
        "output_directory": str(stage), "maximum_updates_per_run": 300000,
        "analysis_and_driver_source_hashes": parent["analysis_and_driver_source_hashes"],
        "extension_source_hashes": {str((protocol / n).relative_to(Path.cwd())): digest(protocol / n) for n in extension_names},
        "recovery_source_hashes": {str(p.relative_to(Path.cwd())) if p.is_absolute() else str(p): digest(p) for p in recovery_paths},
        "recovery_window_steps": 10000, "final_window": [250000, 300000],
        "parent_lower_rate_plan_sha256": digest(parent_path), "parent_source_directory": str(source),
        "parent_complete_archive": str(archive), "parent_complete_comparison": str(comparison),
        "parent_archive_artifact_manifest_sha256": digest(archive / "artifact-hashes.json"),
        "parent_comparison_artifact_manifest_sha256": digest(comparison / "artifact-hashes.json"),
        "parent_checkpoint_sha256": original_hashes["checkpoint.pt"], "parent_run_file_hashes": original_hashes,
        "copied_parent_file_hashes": copied_hashes,
        "CPU_extension_preparation_sha256": digest(protocol / "budget-extension-preparation/validation.json"),
        "budget_exposure": expected_batches(config, 300000),
        "runtime_peak_snapshot_scope": "canonical_observations; original_parent_and_all_observed_extension_segments_maximum; host_recording_cost_in_wall_time",
        "user_direction": "maximum_total_300000_training_updates_per_run; additional_failure_frequency_and_recovery_metrics",
        "scope": "exploratory_posthoc_150000_to_300000_budget_extension; additional_updates_frozen_prospectively; no_relaxed_gate_or_independent_confirmation"})
    write_json(stage / "plan.json", manifest)
    write_json(committed, manifest)
    write_json(protocol / "budget300k-freeze-validation.json", {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(), "frozen_utc": manifest["frozen_utc"],
        "source_commit": manifest["git_commit"], "scientific_training_started": False,
        "completed_parent_updates": 150000, "new_total_steps": 300000, "additional_updates": 150000,
        "changed_configuration_fields": ["steps"], "all_original_checkpoint_and_log_copies_byte_identical": True,
        "native_optimizer_steps_verified": 150000, "original_exposure_and_cursor_verified": True,
        "original_sampling_permutation_verified": True, "original_checkpoint_sha256": original_hashes["checkpoint.pt"],
        "same_environment_training_analysis_corpus_instrumentation_and_criterion": True,
        "CPU_exact_continuation_oracle_passed": True, "recovery_window_steps": 10000,
        "final_window": [250000, 300000], "independent_confirmation_eligible_before_extension": False,
        "scope": "additional_scientific_updates_not_started; actual_guard_checks_required_before_launch"})
    print(json.dumps({"frozen_utc": manifest["frozen_utc"], "stage": str(stage), "total_steps": 300000,
                      "scientific_training_started": False, "checkpoint_sha256": original_hashes["checkpoint.pt"]}))


if __name__ == "__main__":
    main()
