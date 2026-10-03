"""Execute real CPU-only complementary checks with explicitly matched AdamW.

Keep small fixtures distinct from scientific selection, confirmation and paired
architecture results. Preserve raw data, both checkpoints, native optimizer
state, every applied learning rate, complete histories and independent reloads.
"""

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

import torch

from experiments.gpt_mini import GPTMini
from experiments.synthetic_trainers.complementary_config import ComplementaryConfig
from experiments.synthetic_trainers.complementary_report import hashes, verify_run
from experiments.synthetic_trainers.complementary_training import train
from experiments.synthetic_trainers.config import ModelSpec
from experiments.synthetic_trainers.corpus import study_pool
from experiments.synthetic_trainers.data import build_split
from experiments.synthetic_trainers.metrics import evaluate
from experiments.synthetic_trainers.runtime import write_json
from experiments.synthetic_trainers.specs import TaskSpec


def without_times(value):
    if isinstance(value, dict):
        return {key: without_times(item) for key, item in value.items() if key != "generation_seconds"}
    return value


def cases():
    specs = [TaskSpec(task="mqar", length=24, symbols=8, pairs=4, queries=2),
             TaskSpec(task="lookup", length=24, symbols=8, pairs=4, queries=2, hops=1),
             TaskSpec(task="lookup", length=24, symbols=8, pairs=4, queries=2, hops=2),
             TaskSpec(task="copy", length=4, symbols=8),
             TaskSpec(task="parity", length=4),
             TaskSpec(task="parity", length=4, scratchpad="running"),
             TaskSpec(task="crasp", length=8, formula_depth=2, formula_seed=0)]
    config = ComplementaryConfig(steps=16, eval_every=4, batch_size=2,
                                 train_examples=4, validation_examples=4, test_examples=4, target=1.0)
    result = []
    for spec in specs:
        name = f"lookup-hops{spec.hops}" if spec.task == "lookup" else f"{spec.task}-{spec.scratchpad}"
        result.append((name, spec, replace(config, eval_lengths=(spec.length * 2,))))
    result.append(("copy-cosine-tail", specs[3], replace(config, eval_lengths=(8,),
                    learning_rate_schedule="cosine_tail", anneal_start=10, anneal_end=14)))
    return result


def prepare(directory):
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError("Complementary CPU preparation requires a fresh directory")
    directory.mkdir(parents=True)
    model = ModelSpec(width=8, layers=1, heads=1, init_std=.02)
    plan = {"frozen_utc": datetime.now(timezone.utc).isoformat(), "CPU_fixture": True,
            "scientific_run": False, "model": asdict(model),
            "cases": [{"name": name, "task": asdict(spec), "training": asdict(cfg)} for name, spec, cfg in cases()],
            "scope": "real_CPU_implementation_checks_only; no_learning_stability_architecture_or_transfer_claim"}
    write_json(directory / "plan.json", plan)
    rows = []
    source_hashes = None
    for name, spec, config in cases():
        output = directory / name
        result = train(config, spec, model, output)
        current_sources = result["provenance"]["source_hashes"]
        if source_hashes is not None and source_hashes != current_sources:
            raise ValueError("Complementary sources changed between CPU cases")
        source_hashes = current_sources
        pool = study_pool(spec, config)
        splits = {"in_distribution": pool["test"]}
        for length in config.eval_lengths:
            probe = replace(spec, length=length, min_length=None)
            splits[f"length-{length}"] = build_split(probe, "test", config.data_seed, config.test_examples)
        for filename, field in (("final.pt", "test_final"), ("best.pt", "test")):
            loaded = GPTMini(model.reference_config(result["provenance"]["vocab_size"],
                                                    result["provenance"]["context_length"]))
            loaded.load_state_dict(torch.load(output / filename, weights_only=True))
            for label, split in splits.items():
                actual = evaluate(loaded, split.examples, config.batch_size, spec=split.spec)
                if without_times(actual) != without_times(result[field][label]):
                    raise ValueError("Independent checkpoint reload disagrees with complementary scores")
        state = torch.load(output / "optimizer-final.pt", weights_only=True)
        audit = result["complementary_protocol"]["optimizer_audit"]
        if ([int(item["step"].item()) for item in state["state"].values()] != audit["state_steps"]
                or any(value.device.type != "cpu" for item in state["state"].values() for value in item.values())):
            raise ValueError("Stored native optimizer is not the audited CPU state")
        write_json(output / "artifact-hashes.json", {"files": hashes(output)})
        rows.append({"name": name, **verify_run(output),
                     "final_and_selected_checkpoint_predictions_independently_reloaded": True,
                     "native_CPU_optimizer_checkpoint_reloaded": True,
                     "final_ID_sequence_accuracy": result["test_final"]["in_distribution"]["sequence_accuracy"],
                     "selected_ID_sequence_accuracy": result["test"]["in_distribution"]["sequence_accuracy"],
                     "final_length_sequence_accuracy": {key: value["sequence_accuracy"] for key, value in result["test_final"].items()
                                                        if key != "in_distribution"}})
        print(name, "verified", flush=True)
    repository = Path.cwd().resolve()
    snapshots = {}
    for filename, expected in source_hashes.items():
        path = Path(filename).resolve()
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("A complementary source changed before preparation archival")
        if path.is_relative_to(repository):
            relative = path.relative_to(repository)
            if ".venv" in relative.parts:
                continue  # Record the native-library fingerprint without copying a dependency into repository snapshots.
            destination = directory / "sources" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
            snapshots[str(relative)] = expected
    script = Path(__file__).resolve()
    shutil.copyfile(script, directory / "probe-snapshot.py")
    write_json(directory / "validation.json", {"recorded_utc": datetime.now(timezone.utc).isoformat(),
        "CPU_fixture": True, "scientific_run": False, "scientific_architecture_selected": False,
        "scope": plan["scope"], "cases": rows, "total_completed_updates": sum(row["steps_completed"] for row in rows),
        "source_hashes": source_hashes, "repository_source_snapshots": snapshots,
        "probe_script_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
        "novel_ID_meaning": "complete_inputs_absent_from_training_at_training_lengths",
        "transfer_support_scope": "separate_novel_validation_denominators_for_ID_and_each_longer_length",
        "checkpoint_reload_scope": "all_reported_final_and_selected_test_scores; generation_time_excluded"})
    write_json(directory / "artifact-hashes.json", {"files": hashes(directory)})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = prepare(args.output)
    print(json.dumps({"CPU_cases": len(rows), "completed_updates": sum(row["steps_completed"] for row in rows),
                      "scientific_run": False, "torch_imported": "torch" in sys.modules}))


if __name__ == "__main__":
    main()
