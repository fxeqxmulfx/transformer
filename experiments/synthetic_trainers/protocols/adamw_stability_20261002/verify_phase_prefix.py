"""Verify a measured plateau and long confirmation in a partial run, without Torch."""

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path

from experiments.synthetic_trainers.persistence import PersistenceConfig, assess
from experiments.synthetic_trainers.stability_integrity import expected_batches, validate_observation


def verify(directory):
    directory = Path(directory)
    hashes = json.loads((directory / "artifact-hashes.json").read_text())["files"]
    actual = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(directory.iterdir()) if p.is_file() and p.name != "artifact-hashes.json"}
    if actual != hashes:
        raise ValueError("Phase prefix artifact hashes differ")
    plan = json.loads((directory / "plan.json").read_text())
    summary = json.loads((directory / "summary.json").read_text())
    history = [json.loads(line) for line in (directory / "history.jsonl").read_text().splitlines()]
    gradients = [json.loads(line) for line in (directory / "gradients.jsonl").read_text().splitlines()]
    if len(plan["recipes"]) != 1:
        raise ValueError("Phase prefix must identify a single frozen recipe")
    recipe = plan["recipes"][0]
    config, end = recipe["config"], summary["through_update"]
    if (summary["recipe"] != recipe["name"] or summary["config"] != config
            or summary["criterion"] != plan["criterion"]
            or plan["criterion"] != asdict(PersistenceConfig())
            or summary["full_frozen_budget"] != config["steps"]
            or not 0 < end < config["steps"] or end % config["eval_every"]):
        raise ValueError("Phase prefix differs from the full frozen recipe or criterion")
    fingerprints = {**plan["source_hashes"], **plan["analysis_and_driver_source_hashes"], **plan["papers"]}
    if summary["frozen_source_and_paper_hashes"] != fingerprints:
        raise ValueError("Phase prefix source fingerprints differ from its plan")
    if ([p["step"] for p in history] != list(range(0, end + 1, config["eval_every"]))
            or [p["step"] for p in gradients] != list(range(1, end + 1))):
        raise ValueError("Phase prefix observations are missing, duplicated, or out of order")
    previous_training, previous_wall = 0., 0.
    for point in history:
        validate_observation(point, plan, config)
        if point["training_seconds"] < previous_training or point["wall_seconds"] < previous_wall:
            raise ValueError("Phase prefix observation costs are not monotone")
        previous_training, previous_wall = point["training_seconds"], point["wall_seconds"]
        if point["step"]:
            batch = expected_batches(config, point["step"])
            if (point["last_batch_size"] != batch["batch_size"]
                    or not math.isclose(point["epochs_seen"] * plan["corpus"]["train_examples"],
                                        batch["examples_seen"], abs_tol=1e-7)):
                raise ValueError("Phase prefix observation has wrong example exposure")
    for point in gradients:
        batch = expected_batches(config, point["step"])
        rate = config["learning_rate"]
        if config["warmup_steps"]:
            rate *= min(1, (point["step"] - 1) / config["warmup_steps"])
        if (point["batch_size"] != batch["batch_size"] or point["epoch_tail"] != batch["epoch_tail"]
                or point["learning_rate"] != rate or not math.isfinite(point["gradient_l2"])
                or point["gradient_l2"] < 0):
            raise ValueError("Phase prefix gradient has invalid exposure, rate, or norm")
    report = {"plan": {"status": "running", "config": config}, "history": history,
              "completed_steps": end, "final": history[-1]}
    result = assess(report, PersistenceConfig(**plan["criterion"]))
    first_heldout = next((p for p in history if p["heldout"]["accuracy"] >= plan["criterion"]["target"]), None)
    first_train = next((p for p in history if p["train"]["accuracy"] >= plan["criterion"]["target"]), None)
    if (result != summary["assessment"] or first_heldout != summary["first_canonical_heldout_target"]
            or first_train != summary["first_canonical_train_target"]
            or len(history) != summary["history_observations"]
            or len(gradients) != summary["gradient_observations"]
            or max(gradients, key=lambda p: p["gradient_l2"]) != summary["largest_observed_gradient"]):
        raise ValueError("Phase prefix summary differs from its measurements")
    plateau, event = result["plateau"], result["long_confirmation"]
    if (not plateau or not event or event["onset"] <= plateau["end"] or event["confirmed"] != end
            or result["complete_canonical_history"] or result["persistent_final_performance"]
            or result["stable_grokking"]):
        raise ValueError("The prefix must show ordered phases without claiming full persistence")
    return {"verified": True, "through_update": end, "history_observations": len(history),
            "gradient_observations": len(gradients), "plateau": plateau,
            "first_heldout_target": first_heldout["step"], "long_confirmation": event,
            "phase_observed": True, "full_budget_complete": False, "stable_grokking": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", type=Path,
                        default=Path(__file__).parent / "first-long-confirmation-fraction25")
    print(json.dumps(verify(parser.parse_args().directory)))
