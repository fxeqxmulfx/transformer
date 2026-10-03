"""Run eight real CPU normalizer pairs through the native complementary trainer."""

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import torch

from experiments.synthetic_trainers.complementary_attention import softmax_model, sparsemax_model, train
from experiments.synthetic_trainers.complementary_attention_plots import render
from experiments.synthetic_trainers.complementary_attention_report import paired_quality, verify_attention_run
from experiments.synthetic_trainers.complementary_report import hashes
from experiments.synthetic_trainers.complementary_training import optimizer_audit
from experiments.synthetic_trainers.config import ModelSpec
from experiments.synthetic_trainers.corpus import study_pool
from experiments.synthetic_trainers.data import build_split
from experiments.synthetic_trainers.metrics import evaluate
from experiments.synthetic_trainers.runtime import write_json
from .prepare_complementary_native import cases, without_times


def reload_case(path, spec, config, model_spec, result, normalizer):
    factory = {"softmax": softmax_model, "sparsemax": sparsemax_model}[normalizer]
    model = factory(model_spec.reference_config(result["provenance"]["vocab_size"], result["provenance"]["context_length"]))
    pool = study_pool(spec, config)
    splits = {"in_distribution": pool["test"]}
    for length in config.eval_lengths:
        probe = replace(spec, length=length, min_length=None)
        splits[f"length-{length}"] = build_split(probe, "test", config.data_seed, config.test_examples)
    for filename, field in (("final.pt", "test_final"), ("best.pt", "test")):
        model.load_state_dict(torch.load(path / filename, weights_only=True))
        assert all(torch.isfinite(p).all().item() for p in model.parameters())
        for label, split in splits.items():
            actual = evaluate(model, split.examples, config.batch_size, spec=split.spec)
            if without_times(actual) != without_times(result[field][label]):
                raise ValueError("Independent normalizer checkpoint replay disagrees with held-out scores")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
        betas=(.9, .98), eps=1e-8, weight_decay=config.weight_decay)
    state = torch.load(path / "optimizer-final.pt", weights_only=True)
    optimizer.load_state_dict(state)
    actual = json.loads(json.dumps(optimizer_audit(optimizer, config.steps)))
    if actual != json.loads((path / "optimizer-audit.json").read_text()):
        raise ValueError("Actual reloaded native optimizer differs from its recorded audit")
    for parameter, item in optimizer.state.items():
        assert item["exp_avg"].shape == parameter.shape == item["exp_avg_sq"].shape
        assert all(value.device.type == "cpu" for value in item.values())
    return {"final_and_selected_checkpoint_predictions_independently_reloaded": True,
            "native_CPU_optimizer_checkpoint_reloaded": True, "native_state_and_parameters_finite": True,
            "checkpoint_optimizer_audit": actual}


def prepare(directory):
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError("Paired complementary CPU preparation needs a fresh destination")
    if torch.cuda.is_initialized():
        raise RuntimeError("Run this CPU preparation in its own CUDA-hidden subprocess")
    model = ModelSpec(width=8, layers=1, heads=1, init_std=.02)
    pair_specs = cases()
    source = Path(__file__).resolve(); protocol = source.parent
    preparation = {str(path.relative_to(Path.cwd())): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (source, protocol / "verify_complementary_attention.py", protocol / "prepare_complementary_native.py")}
    pairs = [{"name": name, "task": asdict(spec), "training": asdict(config),
              "execution_order": ["softmax", "sparsemax"] if index % 2 == 0 else ["sparsemax", "softmax"]}
             for index, (name, spec, config) in enumerate(pair_specs)]
    plan = {"frozen_utc": datetime.now(timezone.utc).isoformat(), "CPU_fixture": True,
        "scientific_run": False, "scientific_architecture_selected": False,
        "model": asdict(model), "pairs": pairs, "preparation_sources": preparation,
        "scope": "eight_actual_CPU_normalizer_pairs; no_scientific_selection_or_learning_claim"}
    directory.mkdir(parents=True); write_json(directory / "plan.json", plan)
    rows = []; sources = None; outcomes = {}
    for pair, (name, spec, config) in zip(pairs, pair_specs):
        for normalizer in pair["execution_order"]:
            path = directory / "runs" / f"{name}-{normalizer}"
            result = train(config, spec, model, path, attention_normalization=normalizer)
            actual_sources = result["provenance"]["source_hashes"]
            if sources is not None and actual_sources != sources:
                raise ValueError("Complementary sources changed between paired cases")
            sources = actual_sources
            replay = reload_case(path, spec, config, model, result, normalizer)
            write_json(path / "artifact-hashes.json", {"files": hashes(path)})
            row = {"name": path.name, **verify_attention_run(path, attention_normalization=normalizer), **replay}
            rows.append(row); print(path.name, "verified", flush=True)
        outcomes[name] = paired_quality(directory / "runs" / f"{name}-softmax", directory / "runs" / f"{name}-sparsemax")
    repository = Path.cwd().resolve(); snapshots = {}
    for filename, expected in {**sources, **{str(repository / name): value for name, value in preparation.items()}}.items():
        path = Path(filename).resolve()
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("A complementary source changed before archival")
        if path.is_relative_to(repository) and ".venv" not in path.relative_to(repository).parts:
            relative = path.relative_to(repository); destination = directory / "sources" / relative
            destination.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(path, destination)
            snapshots[str(relative)] = expected
    shutil.copyfile(source, directory / "probe-snapshot.py")
    write_json(directory / "summary.json", {"CPU_fixture": True, "scientific_run": False,
        "scientific_architecture_selected": False, "complete_pairs": len(outcomes), "complete_runs": len(rows),
        "total_completed_updates": sum(row["steps_completed"] for row in rows),
        "total_canonical_observations": sum(row["canonical_observations"] for row in rows),
        "pairs": outcomes, "all_failures_retained": True, "plots_included": True})
    write_json(directory / "validation.json", {"recorded_utc": datetime.now(timezone.utc).isoformat(),
        "CPU_fixture": True, "scientific_run": False, "scientific_architecture_selected": False,
        "source_hashes": sources, "repository_source_snapshots": snapshots, "cases": rows,
        "probe_script_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "checkpoint_reload_scope": "all_final_and_selected_ID_and_length_scores; generation_seconds_excluded"})
    render(directory)
    assert not torch.cuda.is_initialized()
    write_json(directory / "artifact-hashes.json", {"files": hashes(directory)})
    return outcomes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    prepare(parser.parse_args().output)
