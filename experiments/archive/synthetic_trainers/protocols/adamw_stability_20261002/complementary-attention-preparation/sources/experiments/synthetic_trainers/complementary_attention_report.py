"""Portable normalizer tags and complete paired complementary quality results.

Checksums and generated-data checks do not independently recompute predictions.
CPU checkpoint replay is recorded separately; scientific gates remain external.
"""

import json
from pathlib import Path

from .complementary_report import verify_run


FACTORIES = {name: f"experiments.synthetic_trainers.complementary_attention.{name}_model"
             for name in ("softmax", "sparsemax")}


def description(normalizer):
    if normalizer not in FACTORIES:
        raise ValueError("Complementary attention requires explicit softmax or sparsemax")
    return {"attention_normalization": normalizer, "factory": FACTORIES[normalizer],
            "changed_fields": ["attention_normalization"],
            "score_scale": "original_GPTMini_scores", "joint_training_convexity_claim": False,
            "scientific_benchmark_and_architecture_gates_are_external": True}


def verify_attention_run(directory, *, attention_normalization):
    """Check the normalizer against a caller's frozen recipe, not a self-label."""
    expected = description(attention_normalization)
    directory = Path(directory)
    outcome = verify_run(directory)
    result = json.loads((directory / "result.json").read_text())
    provenance = result["provenance"]
    if result.get("complementary_attention") != expected or provenance["factory"] != expected["factory"]:
        raise ValueError("Complementary attention differs from the externally declared normalizer")
    files = {str(Path(name)).replace("\\", "/") for name in provenance["source_hashes"]}
    required = ("experiments/gpt_mini.py", "experiments/synthetic_trainers/complementary_attention.py",
                "experiments/synthetic_trainers/complementary_attention_report.py",
                "experiments/synthetic_trainers/complementary_training.py",
                "experiments/synthetic_trainers/complementary_config.py",
                "experiments/synthetic_trainers/sparsemax_attention.py")
    if any(not any(name == path or name.endswith("/" + path) for name in files) for path in required):
        raise ValueError("Complementary provenance omits a normalizer/native training source")
    return {**outcome, "attention_normalization": attention_normalization}


def paired_quality(control, candidate):
    """Keep final/selected and ID/length outcomes separate at identical budgets."""
    paths = {"control": Path(control), "candidate": Path(candidate)}
    summaries = {role: verify_attention_run(path, attention_normalization=norm)
                 for (role, path), norm in zip(paths.items(), ("softmax", "sparsemax"))}
    runs = {role: json.loads((path / "result.json").read_text()) for role, path in paths.items()}
    left, right = runs["control"], runs["candidate"]
    for key in ("training", "task", "model", "source_hashes", "optimizer", "vocab_size", "context_length"):
        if left["provenance"][key] != right["provenance"][key]:
            raise ValueError("Complementary pairs must retain model/data/optimizer/budget/source settings")
    if (left["split_fingerprints"] != right["split_fingerprints"]
            or left["study"]["corpus"] != right["study"]["corpus"]
            or left["parameters"] != right["parameters"]
            or left["examples_seen"] != right["examples_seen"]
            or left["supervised_tokens_seen"] != right["supervised_tokens_seen"]):
        raise ValueError("Complementary pairs must retain exact data, parameter count and exposure")
    outcomes = {}
    for field, label in (("test_final", "final_checkpoint"), ("test", "validation_selected_checkpoint")):
        if left[field].keys() != right[field].keys():
            raise ValueError("Complementary pairs need identical separate ID/length probes")
        outcomes[label] = {name: {"control_sequence_accuracy": left[field][name]["sequence_accuracy"],
            "candidate_sequence_accuracy": right[field][name]["sequence_accuracy"],
            "candidate_minus_control": right[field][name]["sequence_accuracy"] - left[field][name]["sequence_accuracy"]}
            for name in left[field]}
    return {"complete_paired_budgets": True, "summaries": summaries, "quality": outcomes,
            "costs": {role: {key: run[key] for key in ("training_seconds", "training_wall_seconds",
                "total_wall_seconds", "parameters", "peak_cuda_bytes", "time_to_target")}
                for role, run in runs.items()},
            "novel_ID_meaning": "complete_inputs_absent_from_training_at_training_lengths",
            "novel_length_support": {role: summary["novel_length_probe_examples"] for role, summary in summaries.items()},
            "scientific_effect_certified": False,
            "scope": "complete_pair_quality_and_observed_costs; independent_scientific_gates_remain_external"}
