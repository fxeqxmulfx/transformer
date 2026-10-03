"""Verify eight complete CPU complementary pairs without importing PyTorch."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

from experiments.synthetic_trainers.complementary_attention_report import paired_quality, verify_attention_run
from experiments.synthetic_trainers.complementary_report import hashes


def verify(directory):
    directory = Path(directory)
    if hashes(directory) != json.loads((directory / "artifact-hashes.json").read_text())["files"]:
        raise ValueError("Paired complementary preparation artifact hashes differ")
    plan = json.loads((directory / "plan.json").read_text())
    summary = json.loads((directory / "summary.json").read_text())
    validation = json.loads((directory / "validation.json").read_text())
    for record in (plan, summary, validation):
        if (record["CPU_fixture"] is not True or record["scientific_run"] is not False
                or record["scientific_architecture_selected"] is not False):
            raise ValueError("CPU preparation cannot select a scientific architecture")
    if len(plan["pairs"]) != 8 or len({pair["name"] for pair in plan["pairs"]}) != 8:
        raise ValueError("All eight distinct complementary CPU pairs are required")
    cases = {row["name"]: row for row in validation["cases"]}
    expected = {f"{pair['name']}-{norm}" for pair in plan["pairs"] for norm in ("softmax", "sparsemax")}
    if (set(cases) != expected or {p.name for p in (directory / "runs").iterdir() if p.is_dir()} != expected
            or len(validation["cases"]) != 16):
        raise ValueError("Every paired case, including failures, must remain archived")
    total = observations = 0
    derived = {}
    for index, pair in enumerate(plan["pairs"]):
        order = ["softmax", "sparsemax"] if index % 2 == 0 else ["sparsemax", "softmax"]
        if pair["execution_order"] != order:
            raise ValueError("Prospective paired order changed")
        for norm in ("softmax", "sparsemax"):
            name = f"{pair['name']}-{norm}"; path = directory / "runs" / name
            config = json.loads((path / "config.json").read_text())
            if (config["task"] != pair["task"] or config["training"] != pair["training"]
                    or config["model"] != plan["model"] or config["training"]["device"] != "cpu"
                    or config["source_hashes"] != validation["source_hashes"]):
                raise ValueError("Complementary case differs from its frozen complete recipe")
            actual = verify_attention_run(path, attention_normalization=norm)
            stored = cases[name]
            if any(stored[key] != value for key, value in actual.items()):
                raise ValueError("Recorded complementary case verification differs")
            for flag in ("final_and_selected_checkpoint_predictions_independently_reloaded",
                         "native_CPU_optimizer_checkpoint_reloaded", "native_state_and_parameters_finite"):
                if stored[flag] is not True:
                    raise ValueError("The CPU preparation lacks actual independent reloads")
            result = json.loads((path / "result.json").read_text())
            if stored["checkpoint_optimizer_audit"] != result["complementary_protocol"]["optimizer_audit"]:
                raise ValueError("Recorded actual checkpoint audit differs")
            total += actual["steps_completed"]; observations += actual["canonical_observations"]
        derived[pair["name"]] = paired_quality(directory / "runs" / f"{pair['name']}-softmax",
                                              directory / "runs" / f"{pair['name']}-sparsemax")
    if (summary["pairs"] != derived or summary["complete_pairs"] != 8 or summary["complete_runs"] != 16
            or summary["total_completed_updates"] != total or summary["total_canonical_observations"] != observations
            or summary["all_failures_retained"] is not True or summary["plots_included"] is not True):
        raise ValueError("The complete complementary paired summary differs from actual cases")
    for name, expected_hash in validation["repository_source_snapshots"].items():
        if hashlib.sha256((directory / "sources" / name).read_bytes()).hexdigest() != expected_hash:
            raise ValueError("Complementary source snapshot differs")
    if hashlib.sha256((directory / "probe-snapshot.py").read_bytes()).hexdigest() != validation["probe_script_sha256"]:
        raise ValueError("Executed preparation snapshot differs")
    for name, expected_hash in plan["preparation_sources"].items():
        if validation["repository_source_snapshots"].get(name) != expected_hash:
            raise ValueError("A preparation/verifier source was not preserved")
    for figure in ("ID", "length"):
        for extension in ("png", "pdf"):
            if not (directory / "plots" / f"complementary-{figure}.{extension}").is_file():
                raise ValueError("Complementary standalone curves are missing")
    assert "torch" not in sys.modules
    return {"CPU_pairs": 8, "CPU_runs": 16, "completed_updates": total,
            "canonical_observations": observations, "Torch_imported": False, "scientific_run": False,
            "prediction_scope": "recorded_independent_CPU_replays; neural_predictions_not_recomputed_here"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    print(json.dumps(verify(parser.parse_args().directory)))
