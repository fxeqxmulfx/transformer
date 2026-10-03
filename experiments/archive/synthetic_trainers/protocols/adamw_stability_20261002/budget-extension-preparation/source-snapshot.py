"""Check native AdamW budget extension against uninterrupted CPU training.

Convexifying Transformers, Section 4 supplies the task; this is an implementation
oracle, not a scientific learning result. Run with CUDA_VISIBLE_DEVICES=''.
"""

from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import torch

from experiments.synthetic_trainers.paper_reproduction.diagnostics import DiagnosticsConfig
from experiments.synthetic_trainers.paper_reproduction.grokking import RunConfig, train
from experiments.synthetic_trainers.persistence import PersistenceConfig
from experiments.synthetic_trainers.stability import freeze, validate_complete
from experiments.synthetic_trainers.stability_report import save_run


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def equal_state(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(equal_state(left[k], right[k]) for k in left)
    if isinstance(left, (tuple, list)):
        return type(left) is type(right) and len(left) == len(right) and all(equal_state(a, b) for a, b in zip(left, right))
    return left == right


def main():
    torch.set_num_threads(1)
    if torch.cuda.is_available():
        raise ValueError("Hide CUDA for this CPU continuation oracle")
    protocol = Path(__file__).resolve().parent
    parent = json.loads((protocol / "lower-rate-plan.json").read_text())
    destination = protocol / "budget-extension-preparation"
    output = Path("experiments/runs/adamw_stability_20261002/bootstrap/budget_extension_cpu_preparation")
    if destination.exists() or output.exists():
        raise FileExistsError("Budget-extension preparation requires fresh destinations")
    frozen = {name: value for key in ("source_hashes", "analysis_and_driver_source_hashes", "papers")
              for name, value in parent[key].items()}
    assert all(digest(Path(name)) == value for name, value in frozen.items())
    scientific = RunConfig(**parent["recipes"][0]["config"])
    total = replace(scientific, device="cpu", prime=7, batch_size=4, steps=40, eval_every=5)
    initial = replace(total, steps=20)
    diagnostics = DiagnosticsConfig(5, True, True)
    name = "native-adamw-cpu-extension"
    recipe = {"name": name, "config": asdict(total), "changed_mechanism": "CPU_budget_extension_oracle_only"}
    manifest = freeze(output / "continued", [recipe], diagnostics, PersistenceConfig())
    run = output / "continued" / name
    first = train(initial, run, diagnostics=diagnostics)
    original = output / "original20"
    shutil.copytree(run, original)
    original_hashes = {p.name: digest(p) for p in original.iterdir()}
    rejected = []
    for field, value in (("learning_rate", .001), ("weight_decay", .2), ("seed", 1),
                         ("batch_policy", "wrap_epoch"), ("steps", 19)):
        try:
            train(replace(total, **{field: value}), run, resume=True, diagnostics=diagnostics)
        except ValueError as error:
            assert "Resume" in str(error)
            rejected.append(field)
        else:
            raise AssertionError(f"Continuation accepted changed {field}")
        assert {p.name: digest(p) for p in run.iterdir()} == original_hashes
    continued = train(total, run, resume=True, diagnostics=diagnostics)
    fresh_run = output / "uninterrupted40"
    fresh = train(total, fresh_run, diagnostics=diagnostics)
    left = torch.load(run / "checkpoint.pt", weights_only=True, map_location="cpu")
    right = torch.load(fresh_run / "checkpoint.pt", weights_only=True, map_location="cpu")
    state_keys = ("model", "optimizer", "optimizer_steps", "step", "examples_seen",
                  "last_batch_size", "batch_generator_state", "permutation", "cursor")
    assert all(equal_state(left[key], right[key]) for key in state_keys)
    assert all(float(state["step"]) == 40 for state in left["optimizer"]["state"].values())
    assert continued["plan"]["budget_extensions"] == [{"old_steps": 20, "new_steps": 40, "scope": "posthoc_budget_extension"}]
    assert fresh["plan"]["budget_extensions"] == []
    assert [p["step"] for p in continued["history"]] == list(range(0, 41, 5))
    assert len(continued["history"]) == len(fresh["history"])
    for a, b in zip(continued["history"], fresh["history"]):
        assert {k: v for k, v in a.items() if k not in ("training_seconds", "wall_seconds")} == {
            k: v for k, v in b.items() if k not in ("training_seconds", "wall_seconds")}
    logs = ("history.jsonl", "gradients.jsonl", "diagnostics.jsonl", "probes.jsonl")
    for filename in logs:
        assert (run / filename).read_bytes().startswith((original / filename).read_bytes())
    for filename in ("gradients.jsonl", "diagnostics.jsonl"):
        assert (run / filename).read_bytes() == (fresh_run / filename).read_bytes()
    gradients = [json.loads(line) for line in (run / "gradients.jsonl").read_text().splitlines()]
    assert [p["step"] for p in gradients] == list(range(1, 41))
    assert all(p["learning_rate"] == total.learning_rate * min(1, (p["step"] - 1) / 10) for p in gradients)
    assert gradients[20]["learning_rate"] == .0003
    assert continued["training_seconds"] > first["training_seconds"]
    assert continued["diagnostic_seconds"] > first["diagnostic_seconds"]
    assert continued["wall_seconds"] > first["wall_seconds"]
    validate_complete(continued, recipe, manifest)
    save_run(output / "continued", name, output / "portable", render=False)
    code = ("from experiments.synthetic_trainers.stability_report import verify_archive; import sys,json; "
            "verify_archive(sys.argv[1]); print(json.dumps({'verified':True,'torch_imported':'torch' in sys.modules}))")
    verified = json.loads(subprocess.check_output(["/usr/bin/python3", "-c", code, str(output / "portable")], text=True))
    assert verified["verified"] and not verified["torch_imported"]
    assert {p.name: digest(p) for p in original.iterdir()} == original_hashes
    assert all(digest(Path(name)) == value for name, value in frozen.items())
    assert not torch.cuda.is_initialized()
    destination.mkdir()
    record = {"recorded_at_utc": datetime.now(timezone.utc).isoformat(), "source_sha256": digest(Path(__file__)),
        "scientific_configuration_before": asdict(scientific), "scientific_configuration_after": asdict(replace(scientific, steps=300000)),
        "scientific_changed_fields": ["steps"], "cpu_configuration": asdict(total), "cpu_split_update": 20,
        "cpu_total_updates": 40, "full_model_width_retained": True, "matched_checkpoint_fields": list(state_keys),
        "all_native_optimizer_parameter_steps": 40, "complete_canonical_metrics_equal_uninterrupted": True,
        "dense_gradients_and_full_diagnostics_byte_equal_uninterrupted": True, "all_original_log_prefixes_byte_preserved": True,
        "original_checkpoint_sha256": original_hashes["checkpoint.pt"], "rejected_changes": rejected,
        "native_warmup_not_reset": True, "cumulative_training_diagnostic_wall_costs_verified": True,
        "portable_archive_verified_without_torch": True, "cuda_context_created": False,
        "frozen_core_files_unchanged": len(frozen), "scope": "CPU_implementation_oracle; no_scientific_learning_or_stability_claim"}
    (destination / "validation.json").write_text(json.dumps(record, indent=2) + "\n")
    for filename in ("history.jsonl", "gradients.jsonl"):
        (destination / filename).write_bytes((run / filename).read_bytes())
    (destination / "source-snapshot.py").write_bytes(Path(__file__).read_bytes())
    hashes = {p.name: digest(p) for p in destination.iterdir()}
    (destination / "artifact-hashes.json").write_text(json.dumps({"files": hashes}, indent=2) + "\n")
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
