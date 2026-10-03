"""Verify the partial AdamW final-tail failure offline without PyTorch.

Setting: Convexifying Transformers, Section 4. The final-tail criterion is
an explicitly frozen follow-up choice, not a statement from that manuscript.
"""

import hashlib
import json
import math
from pathlib import Path

from experiments.synthetic_trainers.persistence import PersistenceConfig, assess
from experiments.synthetic_trainers.stability_analysis import diagnostic_metrics
from experiments.synthetic_trainers.stability_integrity import expected_batches, validate_observation


def verify(directory):
    directory = Path(directory)
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(directory.iterdir()) if p.is_file() and p.name != "artifact-hashes.json"}
    if files != json.loads((directory / "artifact-hashes.json").read_text())["files"]:
        raise ValueError("Final-tail prefix artifact hashes differ")
    read = lambda name: json.loads((directory / name).read_text())
    plan, summary = read("plan.json"), read("summary.json")
    config = next(r["config"] for r in plan["recipes"] if r["name"] == summary["recipe"])
    criterion = PersistenceConfig(**plan["criterion"])
    if config["optimizer"] != "adamw" or criterion != PersistenceConfig():
        raise ValueError("The prefix differs from the frozen AdamW criterion")
    history = [json.loads(s) for s in (directory / "history.jsonl").read_text().splitlines()]
    gradients = [json.loads(s) for s in (directory / "gradients.jsonl").read_text().splitlines()]
    observations, diagnostics = read("observations.json"), read("diagnostics.json")
    end, cadence = summary["through_update"], config["eval_every"]
    start, stop = end - cadence, end + 1
    indices = [start, start + 1, end - 1, end, stop]
    if (end >= config["steps"] or end < config["steps"] - criterion.tail_steps
            or summary["full_frozen_budget"] != config["steps"]
            or [p["step"] for p in history] != list(range(0, end + 1, cadence))
            or [p["step"] for p in observations] != indices
            or [p["step"] for p in diagnostics] != indices
            or [p["step"] for p in gradients] != list(range(start, stop + 1))):
        raise ValueError("Final-tail prefix coverage or frozen budget differs")
    report = {"plan": {"status": "running", "config": config}, "history": history,
              "completed_steps": end, "final": history[-1]}
    result = assess(report, criterion)
    first = next(p for p in history if p["step"] >= result["tail_start_step"]
                 and min(p["train"]["accuracy"], p["heldout"]["accuracy"]) < criterion.target)
    if (result != summary["assessment"] or first != summary["first_canonical_final_tail_failure"]
            or first != history[-1] or len(history) != summary["history_observations"]
            or summary["gradient_interval_start"] != start or summary["gradient_interval_end"] != stop
            or len(gradients) != summary["gradient_interval_observations"]
            or max(gradients, key=lambda p: p["gradient_l2"]) != summary["largest_gradient_in_interval"]
            or [diagnostic_metrics(p) for p in diagnostics] != summary["derived_diagnostic_metrics"]):
        raise ValueError("Final-tail prefix summary differs from raw measurements")
    if result["complete_canonical_history"] or result["persistent_final_performance"] or result["stable_grokking"]:
        raise ValueError("This prefix cannot certify completion or persistence")
    by_step = {p["step"]: p for p in history}
    previous_training, previous_wall = 0., 0.
    for point in history:
        if point["training_seconds"] < previous_training or point["wall_seconds"] < previous_wall:
            raise ValueError("Canonical measured costs are not monotone")
        previous_training, previous_wall = point["training_seconds"], point["wall_seconds"]
    for point in [*history, *observations]:
        validate_observation(point, plan, config)
        if point["step"] in by_step and point != by_step[point["step"]]:
            raise ValueError("Canonical source samples disagree with their prefix")
        if point["step"]:
            batch = expected_batches(config, point["step"])
            if point["last_batch_size"] != batch["batch_size"] or not math.isclose(
                    point["epochs_seen"] * plan["corpus"]["train_examples"],
                    batch["examples_seen"], abs_tol=1e-7):
                raise ValueError("Observed batch exposure differs from the frozen policy")
    trace = {p["step"]: p for p in gradients}
    for point in gradients:
        batch = expected_batches(config, point["step"])
        rate = config["learning_rate"]
        if config["warmup_steps"]:
            rate *= min(1, (point["step"] - 1) / config["warmup_steps"])
        if (point["batch_size"] != batch["batch_size"] or point["epoch_tail"] != batch["epoch_tail"]
                or point["learning_rate"] != rate
                or not math.isfinite(point["gradient_l2"]) or point["gradient_l2"] < 0):
            raise ValueError("Gradient interval has an invalid norm, rate or batch")
    parameter_names = set(diagnostics[0]["parameters"])
    for point in diagnostics:
        batch = expected_batches(config, point["step"])
        if (not parameter_names or set(point["parameters"]) != parameter_names
                or point["gradient_l2"] != trace[point["step"]]["gradient_l2"]
                or point["learning_rate"] != trace[point["step"]]["learning_rate"]
                or any(point[k] != batch[k] for k in
                       ("batch_size", "cursor_before", "cursor_after", "epoch_tail", "epoch_wraps_in_batch"))):
            raise ValueError("Tensor diagnostics disagree with the dense trace or frozen batch policy")
        required = {"parameter_l2", "parameter_before_l2", "gradient_l2", "update_l2",
                    "exp_avg_l2", "exp_avg_sq_l2"}
        for norms in point["parameters"].values():
            if not required <= norms.keys() or any(not math.isfinite(v) or v < 0 for v in norms.values()):
                raise ValueError("Invalid or incomplete native AdamW diagnostic norms")
        joint = math.sqrt(sum(p["gradient_l2"] ** 2 for p in point["parameters"].values()))
        if not math.isclose(joint, point["gradient_l2"], rel_tol=1e-5, abs_tol=1e-8):
            raise ValueError("Joint gradient differs from the full tensor samples")
        if any(not math.isfinite(point[k]) or point[k] < 0 for k in ("answer_loss", "EOS_loss")):
            raise ValueError("Invalid sampled minibatch loss")
        for values in point["temperatures"].values():
            if len(values["log_alpha"]) != len(values["inverse_temperature"]):
                raise ValueError("Temperature support differs")
            for logarithm, temperature in zip(values["log_alpha"], values["inverse_temperature"]):
                if not math.isfinite(logarithm) or not math.isclose(math.exp(logarithm), temperature, rel_tol=1e-6):
                    raise ValueError("Invalid sampled inverse temperature")
    return {"verified": True, "through_update": end, "history_observations": len(history),
            "gradient_interval_observations": len(gradients), "full_tensor_samples": len(diagnostics),
            "first_final_tail_failure": first["step"], "frozen_persistence_ruled_out": True,
            "complete_budget": False, "causal_trigger_identified": False}


if __name__ == "__main__":
    print(json.dumps(verify(Path(__file__).parent / "first-tail-failure")))
