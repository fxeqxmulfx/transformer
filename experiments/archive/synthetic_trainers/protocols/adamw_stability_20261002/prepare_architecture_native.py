"""Execute two complete CPU normalizer pairs with identical native schedules.

This is implementation preparation. Neither passing CPU checks nor a single
exploratory GPU pair substitutes for all six scientific primary confirmations.
"""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path

from experiments.synthetic_trainers import architecture_report, architecture_training, stability_report
from experiments.synthetic_trainers.attention_layout import digest
from experiments.synthetic_trainers.paper_reproduction.diagnostics import DiagnosticsConfig
from experiments.synthetic_trainers.paper_reproduction.modular_data import make_corpus
from experiments.synthetic_trainers.persistence import PersistenceConfig
from experiments.synthetic_trainers.stability_protocol import environment, paper_fingerprints
from experiments.synthetic_trainers.stability_recovery import describe
from experiments.synthetic_trainers.stability_recovery_series import timeline
from experiments.synthetic_trainers.tagged_confirmation_layout import current_sources
from experiments.synthetic_trainers.tagged_confirmation_report import verified_benchmark


PROTOCOL = Path("experiments/synthetic_trainers/protocols/adamw_stability_20261002")


def source_hashes():
    result = current_sources()
    for path in (Path("experiments/synthetic_trainers/architecture_training.py"),
                 Path("experiments/synthetic_trainers/architecture_report.py"), Path(__file__).relative_to(Path.cwd()),
                 PROTOCOL / "verify_architecture_native.py"):
        result[str(path)] = digest(path)
    return dict(sorted(result.items()))


def prepare(stage, destination):
    stage, destination = Path(stage), Path(destination)
    if stage.exists() or destination.exists():
        raise FileExistsError("Architecture CPU preparation requires fresh raw and archive directories")
    config = dict(model="gptmini", optimizer="adamw", prime=7, train_fraction=.5, steps=40,
                  batch_size=4, eval_every=5, learning_rate=.0003, weight_decay=.1,
                  anneal_start=20, anneal_end=30, device="cpu")
    recipes = [{"name": f"{schedule}-{norm}", "config": asdict(architecture_training.ArchitectureRunConfig(
                **config, learning_rate_schedule=schedule, attention_normalization=norm))}
                for schedule in ("constant", "cosine_tail") for norm in ("sparsemax", "softmax")]
    criterion = PersistenceConfig(plateau_steps=5, plateau_observations=2,
                                  confirmation_observations=2, tail_steps=10)
    diagnostics = DiagnosticsConfig(5, True, True)
    active = json.loads((PROTOCOL / "attention-pair-plan.json").read_text())
    manifest = {"frozen_utc": datetime.now(timezone.utc).isoformat(), "stage": "CPU_native_architecture_preparation",
        "scientific_run": False, "CPU_fixture": True, "campaign_architecture_claim_allowed": False,
        "recipes": recipes, "planned_runs": 4, "execution_order": [recipe["name"] for recipe in recipes],
        "maximum_updates_per_run": 300000, "criterion": asdict(criterion), "instrumentation": asdict(diagnostics),
        "training_source_hashes": architecture_training.training_sources(), "source_hashes": source_hashes(),
        "papers": paper_fingerprints(), "environment": environment("cpu"), "corpus": make_corpus(7, .5, 0).summary(),
        "Lean_specification_hashes": active["Lean_specification_hashes"],
        "Lean_audit_validation_sha256": active["Lean_audit_validation_sha256"],
        "scope": "real_CPU_native_implementation_pairs; no_scientific_selection_learning_or_architecture_claim"}
    for name, expected in manifest["source_hashes"].items():
        if digest(Path(name)) != expected:
            raise ValueError("An architecture CPU source changed before the first update")
    stage.mkdir(parents=True)
    stability_report.write_json(stage / "plan.json", manifest)
    destination.mkdir(parents=True)
    stability_report.write_json(destination / "plan.json", manifest)
    summaries, recovery, checkpoints = {}, {}, {}
    for recipe in recipes:
        name = recipe["name"]
        for filename, expected in manifest["source_hashes"].items():
            if digest(Path(filename)) != expected:
                raise ValueError("An architecture CPU source changed between planned cases")
        measurements = architecture_training.train(architecture_training.ArchitectureRunConfig(**recipe["config"]),
                                                   stage / name, diagnostics=diagnostics)
        output = destination / name
        summaries[name] = architecture_report.save_run(stage, name, output, render=True)
        checkpoints[name] = {"raw_checkpoint_sha256": digest(stage / name / "checkpoint.pt"),
                             "portable_scope": "native_checkpoint_local; raw_histories_and_checksums_archived"}
        recovery[name] = {"tail_and_episodes": describe(measurements, criterion, window_steps=10),
                          "whole_post_onset_series": timeline(measurements, criterion, window_steps=10)}
        print(name, "complete_archive_verified", flush=True)
    pairs = {schedule: architecture_report.paired_normalizer_outcome(
             summaries[f"{schedule}-softmax"], summaries[f"{schedule}-sparsemax"])
             for schedule in ("constant", "cosine_tail")}
    guards = {}
    for mode in ("normalizer", "schedule"):
        try:
            verified_benchmark(PROTOCOL / "tagged-confirmation-preparation" / mode)
        except ValueError as error:
            guards[mode] = str(error)
        else:
            raise ValueError("CPU confirmation unexpectedly opened the scientific benchmark gate")
    stability_report.write_json(destination / "architecture-pairs.json", pairs)
    stability_report.write_json(destination / "recovery.json", recovery)
    stability_report.write_json(destination / "checkpoints.json", checkpoints)
    stability_report.write_json(destination / "validation.json", {"recorded_utc": datetime.now(timezone.utc).isoformat(),
        "CPU_fixture": True, "scientific_run": False, "scientific_architecture_selected": False,
        "cases": [{"name": name, "parameters": summary["parameters"], "completed_updates": summary["config"]["steps"],
                   "stable_grokking": summary["assessment"]["stable_grokking"],
                   "canonical_observations": summary["canonical_observations"],
                   "gradient_observations": summary["gradient_trace"]["observations"]} for name, summary in summaries.items()],
        "total_completed_updates": sum(summary["config"]["steps"] for summary in summaries.values()),
        "scientific_benchmark_guard_rejections": guards})
    stability_report.write_json(destination / "artifact-hashes.json", {"files":
        {str(p.relative_to(destination)): digest(p) for p in sorted(destination.rglob("*"))
         if p.is_file() and p != destination / "artifact-hashes.json"}})
    return pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    pairs = prepare(args.stage, args.archive)
    print(json.dumps({"CPU_pairs": len(pairs), "completed_updates": 160, "scientific_run": False,
                      "eligible_pair_support": {name: row["eligible_pair_support"] for name, row in pairs.items()}}))


if __name__ == "__main__":
    main()
