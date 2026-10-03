"""Verify the complete parent and frozen 300,000-update extension without PyTorch.

Convexifying Transformers, Section 4: this exploratory continuation preserves
the original failed budget and fixes all additional updates prospectively.
"""

import hashlib
import json
from pathlib import Path

from experiments.synthetic_trainers.stability_comparison import verify_comparison
from experiments.synthetic_trainers.stability_report import verify_archive


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_frozen(protocol=None):
    protocol = Path(__file__).resolve().parent if protocol is None else Path(protocol)
    expected = json.loads((protocol / "budget300k-plan.json").read_text())
    stage = Path(expected["output_directory"])
    if json.loads((stage / "plan.json").read_text()) != expected:
        raise ValueError("Live budget-extension plan differs from its committed manifest")
    parent_path = protocol / "lower-rate-plan.json"
    if digest(parent_path) != expected["parent_lower_rate_plan_sha256"]:
        raise ValueError("Original lower-rate plan changed")
    parent = json.loads(parent_path.read_text())
    if (len(expected["recipes"]) != 1 or expected["criterion"] != parent["criterion"]
            or expected["maximum_updates_per_run"] != 300000):
        raise ValueError("Budget extension changed the criterion, recipe count or user cap")
    config, original = expected["recipes"][0]["config"], parent["recipes"][0]["config"]
    if (set(config) != set(original) or {key for key in config if config[key] != original[key]} != {"steps"}
            or original["steps"] != 150000 or config["steps"] != 300000 or config["optimizer"] != "adamw"):
        raise ValueError("The continuation may only extend total steps from 150,000 to 300,000")
    for key in ("source_hashes", "training_source_hashes", "analysis_and_driver_source_hashes",
                "instrumentation", "environment", "papers", "corpus"):
        if expected[key] != parent[key]:
            raise ValueError(f"Budget extension changed parent {key}")
    prefix = "experiments/synthetic_trainers/protocols/adamw_stability_20261002/"
    if set(expected["extension_source_hashes"]) != {prefix + name for name in (
            "budget_extension_guard.py", "run_budget_extension.py", "archive_budget_extension.py",
            "prepare_budget_extension.py", "freeze_budget_extension.py")}:
        raise ValueError("The complete extension driver source list must remain pinned")
    if set(expected["recovery_source_hashes"]) != {
            "experiments/synthetic_trainers/stability_recovery.py",
            "experiments/synthetic_trainers/stability_recovery_series.py", prefix + "plot_recovery.py"}:
        raise ValueError("All recovery analysis sources must remain pinned")
    for key in ("source_hashes", "analysis_and_driver_source_hashes", "papers", "extension_source_hashes", "recovery_source_hashes"):
        for name, value in expected[key].items():
            if digest(name) != value:
                raise ValueError(f"Frozen budget-extension source changed: {name}")
    if expected["recovery_window_steps"] != 10000 or expected["final_window"] != [250000, 300000]:
        raise ValueError("Frozen recovery width or final persistence window changed")
    preparation = protocol / "budget-extension-preparation/validation.json"
    if digest(preparation) != expected["CPU_extension_preparation_sha256"]:
        raise ValueError("Verified native continuation preparation changed")
    prepared = json.loads(preparation.read_text())
    if (prepared["scientific_configuration_before"] != original or prepared["scientific_configuration_after"] != config
            or not prepared["dense_gradients_and_full_diagnostics_byte_equal_uninterrupted"]
            or not prepared["native_warmup_not_reset"]):
        raise ValueError("CPU oracle does not verify this exact budget intervention")
    archive, comparison = Path(expected["parent_complete_archive"]), Path(expected["parent_complete_comparison"])
    for path, key in ((archive, "parent_archive_artifact_manifest_sha256"),
                      (comparison, "parent_comparison_artifact_manifest_sha256")):
        if digest(path / "artifact-hashes.json") != expected[key]:
            raise ValueError("Complete parent artifact manifest changed")
    summary, combined = verify_archive(archive), verify_comparison(comparison)
    assessment = summary["assessment"]
    if (not combined["complete_stage"] or combined["completed_runs"] != 1
            or combined["ready_to_freeze_independent_confirmation"] or not assessment["complete_canonical_history"]
            or not assessment["plateau"] or not assessment["long_confirmation"]
            or assessment["persistent_final_performance"] or not assessment["tail_failures"]):
        raise ValueError("Continuation requires the complete phase-eligible, persistence-failing parent")
    if json.loads((comparison / "plan.json").read_text()) != parent:
        raise ValueError("Complete parent differs from its original frozen plan")
    source = Path(expected["parent_source_directory"])
    for filename, value in expected["parent_run_file_hashes"].items():
        if digest(source / filename) != value:
            raise ValueError(f"Original 150,000-update resource changed: {filename}")
    if digest(source / "checkpoint.pt") != expected["parent_checkpoint_sha256"]:
        raise ValueError("Original native checkpoint changed")
    if (source / "measurements.json").read_bytes() != (archive / "measurements.json").read_bytes():
        raise ValueError("Local parent measurement differs from its complete archive")
    return stage, expected


def verify_prefixes(stage, manifest):
    run = stage / manifest["recipes"][0]["name"]
    original = Path(manifest["parent_source_directory"])
    for filename in ("history.jsonl", "gradients.jsonl", "diagnostics.jsonl", "probes.jsonl"):
        if not (run / filename).read_bytes().startswith((original / filename).read_bytes()):
            raise ValueError(f"Continuation changed or omitted original {filename} observations")


def verify_child(stage, manifest):
    run = stage / manifest["recipes"][0]["name"]
    plan = json.loads((run / "plan.json").read_text())
    budget = plan["config"]["steps"]
    if budget == 150000:
        for filename, value in manifest["copied_parent_file_hashes"].items():
            if digest(run / filename) != value:
                raise ValueError(f"Initial continuation copy changed: {filename}")
        if (run / "measurements.json").exists():
            raise ValueError("Old complete measurements must remain solely in the original parent")
    elif budget == 300000:
        expected_extension = [{"old_steps": 150000, "new_steps": 300000, "scope": "posthoc_budget_extension"}]
        if plan["config"] != manifest["recipes"][0]["config"] or plan["budget_extensions"] != expected_extension:
            raise ValueError("Resumed extension config or provenance changed")
        for key, expected in (("source_hashes", manifest["training_source_hashes"]),
                              ("instrumentation", manifest["instrumentation"]), ("corpus", manifest["corpus"]),
                              ("torch", manifest["environment"]["torch"]), ("gpu", manifest["environment"]["gpu"])):
            if plan[key] != expected:
                raise ValueError(f"Resumed extension {key} changed")
    else:
        raise ValueError("Unexpected checkpoint plan budget")
    verify_prefixes(stage, manifest)
    return run
