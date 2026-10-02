"""Preserve the exact first ordered-phase prefix of a frozen calibration.

Convexifying Transformers, Section 4: the stronger phase criterion is the
repository's explicit adaptation. A partial prefix never certifies persistence.
"""

import argparse
import hashlib
import json
from pathlib import Path
import tempfile

from experiments.synthetic_trainers.persistence import PersistenceConfig, assess
from .verify_phase_prefix import verify


def complete_lines(path):
    data = path.read_bytes()
    return data[:data.rfind(b"\n") + 1].splitlines(keepends=True)


def record(stage, destination):
    stage, destination = Path(stage), Path(destination)
    if destination.exists():
        raise FileExistsError("Phase prefixes require a fresh destination")
    plan_bytes = (stage / "plan.json").read_bytes()
    plan = json.loads(plan_bytes)
    if len(plan["recipes"]) != 1:
        raise ValueError("Identify a single frozen calibration recipe")
    recipe = plan["recipes"][0]
    config, criterion = recipe["config"], plan["criterion"]
    fingerprints = {**plan["source_hashes"], **plan["analysis_and_driver_source_hashes"], **plan["papers"]}
    for name, digest in fingerprints.items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
            raise ValueError("Frozen calibration source or manuscript changed")
    source = stage / recipe["name"]
    lines = complete_lines(source / "history.jsonl")
    history = [json.loads(line) for line in lines]

    def assessment(points):
        report = {"plan": {"status": "running", "config": config}, "history": points,
                  "completed_steps": points[-1]["step"], "final": points[-1]}
        return assess(report, PersistenceConfig(**criterion))

    result = assessment(history)
    if not result["plateau"] or not result["long_confirmation"]:
        raise ValueError("No qualifying plateau followed by long confirmation has been observed")
    end = result["long_confirmation"]["confirmed"]
    lines = [line for line in lines if json.loads(line)["step"] <= end]
    history = [json.loads(line) for line in lines]
    gradient_lines = [line for line in complete_lines(source / "gradients.jsonl")
                      if json.loads(line)["step"] <= end]
    gradients = [json.loads(line) for line in gradient_lines]
    summary = {"recipe": recipe["name"], "through_update": end,
               "full_frozen_budget": config["steps"], "config": config, "criterion": criterion,
               "assessment": assessment(history),
               "first_canonical_heldout_target": next(p for p in history if p["heldout"]["accuracy"] >= criterion["target"]),
               "first_canonical_train_target": next(p for p in history if p["train"]["accuracy"] >= criterion["target"]),
               "history_observations": len(history), "gradient_observations": len(gradients),
               "largest_observed_gradient": max(gradients, key=lambda p: p["gradient_l2"]),
               "frozen_source_and_paper_hashes": fingerprints,
               "scope": "complete_canonical_and_gradient_prefix_through_first_long_confirmation; "
                        "observed_delayed_generalization; incomplete_budget_and_unresolved_persistence_and_repeatability"}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="phase-prefix-", dir=destination.parent) as temporary:
        output = Path(temporary) / "prefix"
        output.mkdir()
        (output / "plan.json").write_bytes(plan_bytes)
        (output / "history.jsonl").write_bytes(b"".join(lines))
        (output / "gradients.jsonl").write_bytes(b"".join(gradient_lines))
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output.iterdir())}
        (output / "artifact-hashes.json").write_text(json.dumps({"files": hashes}, indent=2) + "\n")
        verification = verify(output)
        if (stage / "plan.json").read_bytes() != plan_bytes:
            raise ValueError("Live plan changed during prefix capture")
        output.rename(destination)
    return verification


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(json.dumps(record(args.stage, args.destination)))
