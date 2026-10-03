"""Portable integrity checks for explicit complementary native AdamW studies.

These checks retain complete budgets, actual rates, checkpoint files and
separate transfer support. They cannot certify a scientific benchmark or
independently recompute neural predictions without loading the checkpoints.
"""

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

from .complementary_config import ComplementaryConfig, learning_rate, optimizer_description
from .corpus import corpus_report, problem_key, study_pool
from .data import GENERATOR_VERSION, build_split
from .specs import TaskSpec


def hashes(directory):
    directory = Path(directory)
    return {str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(directory.rglob("*"))
            if path.is_file() and path != directory / "artifact-hashes.json"}


def verify_run(directory):
    """Validate raw data against deterministic generators, never infer transfer."""
    directory = Path(directory)
    stored = json.loads((directory / "artifact-hashes.json").read_text())["files"]
    if stored != hashes(directory):
        raise ValueError("Complementary archive file checksums differ")
    result = json.loads((directory / "result.json").read_text())
    provenance = json.loads((directory / "config.json").read_text())
    if result["provenance"] != provenance:
        raise ValueError("Complementary provenance differs from the actual run")
    fields = provenance["training"]
    config = ComplementaryConfig(**(fields | {"eval_lengths": tuple(fields["eval_lengths"])}))
    spec = TaskSpec(**provenance["task"])
    if (spec.task not in ("mqar", "lookup", "copy", "parity", "crasp")
            or provenance["model"]["init_std"] != .02 or provenance["optimizer"] != optimizer_description(config)):
        raise ValueError("Complementary adaptation has different model initialization or optimizer semantics")
    if result["steps_completed"] != config.steps or result["complementary_protocol"]["config"] != fields:
        raise ValueError("Complementary archive is not a complete tagged budget")
    for filename in ("best.pt", "final.pt", "optimizer-final.pt"):
        if not (directory / filename).is_file():
            raise ValueError("Complementary archive lacks a required checkpoint")
    rates = [json.loads(line) for line in (directory / "update-rates.jsonl").read_text().splitlines()]
    expected = [{"step": step, "learning_rates": [learning_rate(config, step - 1)],
                 "weight_decays": [config.weight_decay]} for step in range(1, config.steps + 1)]
    if rates != expected:
        raise ValueError("Complementary native learning-rate history differs from the fixed schedule")
    audit = json.loads((directory / "optimizer-audit.json").read_text())
    groups = [{"lr": learning_rate(config, config.steps - 1), "weight_decay": config.weight_decay,
               "betas": [.9, .98], "eps": 1e-8, "amsgrad": False,
               "parameter_tensors": audit["parameter_tensors"], "parameter_scalars": result["parameters"]}]
    if (audit != result["complementary_protocol"]["optimizer_audit"]
            or audit["native_class"] != "torch.optim.adamw.AdamW"
            or audit["completed_updates"] != config.steps or audit["parameter_scalars"] != result["parameters"]
            or audit["state_steps"] != [config.steps] * audit["parameter_tensors"]
            or audit["state_keys"] != [["exp_avg", "exp_avg_sq", "step"]] * audit["parameter_tensors"]
            or not audit["all_state_tensors_finite"] or audit["groups"] != groups):
        raise ValueError("Complementary native optimizer audit differs from the actual recipe")
    history = [json.loads(line) for line in (directory / "history.jsonl").read_text().splitlines()]
    steps = sorted({0, config.steps, *range(config.eval_every, config.steps + 1, config.eval_every)})
    if [row["step"] for row in history] != steps:
        raise ValueError("Complementary archive lacks the complete canonical history")
    pool = study_pool(spec, config)
    transfer = {f"length-{length}": replace(spec, length=length, min_length=None) for length in config.eval_lengths}
    datasets = {"train": pool["train"], "train_clean": pool["train"], "validation": pool["validation"],
                "in_distribution": pool["test"]}
    for name, probe_spec in transfer.items():
        datasets[name] = build_split(probe_spec, "test", config.data_seed, config.test_examples)
        datasets[f"validation_probes/{name}"] = build_split(probe_spec, "validation", config.data_seed, config.validation_examples)
    for name, split in datasets.items():
        data = directory / "data" / name
        actual = [json.loads(line) for line in (data / "examples.jsonl").read_text().splitlines()]
        expected_rows = json.loads(json.dumps([asdict(row) for row in split.examples]))
        metadata = json.loads((data / "metadata.json").read_text())
        expected_metadata = {"version": GENERATOR_VERSION, "name": split.name, "spec": split.spec.sampling_spec(),
                             "seed": split.seed, "count": len(split.examples), "fingerprint": split.fingerprint}
        if actual != expected_rows or metadata != expected_metadata:
            raise ValueError("Complementary dataset differs from the frozen deterministic generator")
    expected_fingerprints = {name: split.fingerprint for name, split in datasets.items()
                             if name != "train_clean" and not name.startswith("validation_probes/")}
    expected_fingerprints["train_clean"] = pool["train"].fingerprint
    if result["split_fingerprints"] != expected_fingerprints:
        raise ValueError("Complementary result fingerprints differ from its archived data")
    if result["study"]["corpus"] != corpus_report(pool):
        raise ValueError("Complementary split overlap diagnostics differ from the actual data")
    seen = {problem_key(row) for row in pool["train"].examples}
    support = result["study"]["generalization_transition"]
    novelty = {name: sum(problem_key(row) not in seen for row in datasets[f"validation_probes/{name}"].examples)
               for name in transfer}
    if support["novel_validation_examples"] != config.validation_examples or support["novel_probe_examples"] != novelty:
        raise ValueError("Complementary novel-ID or length support differs from the actual data")
    if any(set(result[field]) != {"in_distribution", *transfer} for field in ("test", "test_final")):
        raise ValueError("Complementary final and selected checkpoint results lack separate transfer splits")
    return {"steps_completed": config.steps, "canonical_observations": len(history),
            "parameters": result["parameters"], "novel_ID_validation_examples": config.validation_examples,
            "novel_length_probe_examples": novelty, "actual_rate_observations": len(rates),
            "scientific_benchmark_certified": False,
            "verification_scope": "checksums_rates_generated_data_and_support; predictions_require_checkpoint_reload"}
