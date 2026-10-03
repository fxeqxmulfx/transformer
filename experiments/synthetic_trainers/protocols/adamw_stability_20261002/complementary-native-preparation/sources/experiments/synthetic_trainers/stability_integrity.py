"""Offline checks for complete modular measurements and diagnostic coverage."""

import math

from .persistence import PersistenceConfig, assess


def expected_batches(config, step):
    """Independent arithmetic check of batch sizes and shuffled-epoch cursors."""
    size = round(config["prime"] * (config["prime"] - 1) * config["train_fraction"])
    batch = config["batch_size"]
    if config["batch_policy"] not in ("short_final", "wrap_epoch"):
        raise ValueError("Unknown declared batch policy")
    if config["batch_policy"] == "short_final":
        updates = math.ceil(size / batch)
        start = (step - 1) % updates * batch
        count, wraps = min(batch, size - start), 0
        end = start + count
        epochs, partial = divmod(step, updates)
        seen = epochs * size + min(partial * batch, size)
    else:
        start = (step - 1) * batch % size
        count, wraps = batch, (start + batch - 1) // size
        end = (start + batch - 1) % size + 1
        seen = step * batch
    return {"batch_size": count, "cursor_before": start, "cursor_after": end,
            "epoch_tail": end == size, "epoch_wraps_in_batch": wraps, "examples_seen": seen}


def validate_report(report, recipe, manifest):
    plan = report["plan"]
    expected = {"config": recipe["config"], "source_hashes": manifest["training_source_hashes"],
                "instrumentation": manifest["instrumentation"], "corpus": manifest["corpus"],
                "torch": manifest["environment"]["torch"], "gpu": manifest["environment"]["gpu"]}
    for key, value in expected.items():
        if plan.get(key) != value:
            raise ValueError(f"Run {key} differs from the frozen protocol")
    assessment = assess(report, PersistenceConfig(**manifest["criterion"]))
    if not assessment["complete_canonical_history"]:
        raise ValueError("Only a complete canonical history can be archived")
    previous_training, previous_wall = 0., 0.
    for point in report["history"]:
        validate_observation(point, manifest, report["plan"]["config"])
        if point["training_seconds"] < previous_training or point["wall_seconds"] < previous_wall:
            raise ValueError("Observation costs are not monotone")
        previous_training, previous_wall = point["training_seconds"], point["wall_seconds"]
    return assessment


def validate_observation(point, manifest, config):
    for split, count in (("train", manifest["corpus"]["train_examples"]),
                         ("heldout", manifest["corpus"]["heldout_examples"])):
        scores = point[split]
        if scores["examples"] != count:
            raise ValueError("Observation lacks the exhaustive split")
        for key in ("accuracy", "answer_accuracy", "EOS_accuracy"):
            if not math.isfinite(scores[key]) or not 0 <= scores[key] <= 1:
                raise ValueError("Invalid observed accuracy")
        for key in ("loss", "answer_loss", "EOS_loss"):
            if not math.isfinite(scores[key]) or scores[key] < 0:
                raise ValueError("Invalid observed loss")
        if scores["accuracy"] > min(scores["answer_accuracy"], scores["EOS_accuracy"]) + 1e-12:
            raise ValueError("Complete RHS accuracy exceeds a component")
        if scores["accuracy"] < scores["answer_accuracy"] + scores["EOS_accuracy"] - 1 - 1e-12:
            raise ValueError("Complete RHS accuracy violates the intersection bound")
        if not math.isclose(2 * scores["loss"], scores["answer_loss"] + scores["EOS_loss"],
                            rel_tol=1e-6, abs_tol=1e-7):
            raise ValueError("RHS loss does not agree with component losses")
    if not 0 <= point["step"] <= config["steps"]:
        raise ValueError("Observation lies outside the frozen budget")
    if any(not math.isfinite(point[key]) or point[key] < 0 for key in ("training_seconds", "wall_seconds")):
        raise ValueError("Invalid measured cost")


def validate_logs(report, diagnostics, probes, manifest, gradients=()):
    config, instrumentation = report["plan"]["config"], manifest["instrumentation"]
    budget, cadence, every = config["steps"], config["eval_every"], instrumentation["every"]
    neighbors = instrumentation["eval_neighbors"]
    validate_gradient_trace(report, diagnostics, gradients, manifest)
    expected_diagnostics = [step for step in range(1, budget + 1)
        if every and step % every == 0 or neighbors and step % cadence in (1, cadence - 1)]
    expected_probes = [step for step in range(1, budget)
        if neighbors and step % cadence in (1, cadence - 1) and step % cadence != 0]
    if [row["step"] for row in diagnostics] != expected_diagnostics:
        raise ValueError("Diagnostic observations are missing, duplicated, or unscheduled")
    if [row["step"] for row in probes] != expected_probes:
        raise ValueError("Neighbor observations are missing, duplicated, or unscheduled")
    for point in [*report["history"], *probes]:
        validate_observation(point, manifest, config)
        if point["step"]:
            batches = expected_batches(config, point["step"])
            if point["last_batch_size"] != batches["batch_size"]:
                raise ValueError("Evaluation batch exposure differs from the declared policy")
            if not math.isclose(point["epochs_seen"] * manifest["corpus"]["train_examples"],
                                batches["examples_seen"], abs_tol=1e-7):
                raise ValueError("Observed example exposure differs from the declared policy")
    parameter_names = set(diagnostics[0]["parameters"]) if diagnostics else set()
    required_norms = {"parameter_l2", "parameter_before_l2", "gradient_l2", "update_l2"}
    if config["optimizer"] == "amsgradw":
        required_norms |= {"m_l2", "v_l2", "maximum_l2", "maximum_min", "maximum_max"}
    else:
        required_norms |= {"exp_avg_l2", "exp_avg_sq_l2"}
    for row in diagnostics:
        expected = expected_batches(config, row["step"])
        for key in ("batch_size", "cursor_before", "cursor_after", "epoch_tail", "epoch_wraps_in_batch"):
            if row[key] != expected[key]:
                raise ValueError(f"Diagnostic {key} differs from the declared batch policy")
        if set(row["parameters"]) != parameter_names:
            raise ValueError("Diagnostic parameter support changed")
        if not parameter_names or any(not required_norms <= norms.keys() for norms in row["parameters"].values()):
            raise ValueError("Diagnostic optimizer moment or parameter norms are missing")
        for norms in row["parameters"].values():
            if config["optimizer"] == "amsgradw" and not 0 <= norms["maximum_min"] <= norms["maximum_max"]:
                raise ValueError("Invalid maximum second-moment range")
        rate = config["learning_rate"]
        if config["warmup_steps"]:
            rate *= min(1, (row["step"] - 1) / config["warmup_steps"])
        if not math.isclose(row["learning_rate"], rate, rel_tol=1e-12, abs_tol=1e-15):
            raise ValueError("Diagnostic learning rate differs from the frozen schedule")
        if any(not math.isfinite(value) or value < 0 for value in
               [row["gradient_l2"], row["answer_loss"], row["EOS_loss"],
                *(value for norms in row["parameters"].values() for value in norms.values())]):
            raise ValueError("Invalid diagnostic norm or loss")
        joint_gradient = math.sqrt(sum(norms["gradient_l2"] ** 2 for norms in row["parameters"].values()))
        if not math.isclose(joint_gradient, row["gradient_l2"], rel_tol=1e-5, abs_tol=1e-8):
            raise ValueError("Joint gradient disagrees with per-tensor norms")
        for values in row["temperatures"].values():
            if len(values["log_alpha"]) != len(values["inverse_temperature"]):
                raise ValueError("Temperature support differs")
            for logarithm, temperature in zip(values["log_alpha"], values["inverse_temperature"]):
                if not math.isfinite(logarithm) or not math.isclose(math.exp(logarithm), temperature, rel_tol=1e-6):
                    raise ValueError("Invalid inverse-temperature diagnostic")


def validate_gradient_trace(report, diagnostics, gradients, manifest):
    """Check every recorded update and agreement with full tensor diagnostics."""
    config = report["plan"]["config"]
    enabled = manifest["instrumentation"].get("trace_gradients", False)
    expected = list(range(1, config["steps"] + 1)) if enabled else []
    if [row["step"] for row in gradients] != expected:
        raise ValueError("Gradient trace is missing, duplicated, or undeclared")
    sampled = {row["step"]: row["gradient_l2"] for row in diagnostics}
    for row in gradients:
        batch = expected_batches(config, row["step"])
        if row["batch_size"] != batch["batch_size"] or row["epoch_tail"] != batch["epoch_tail"]:
            raise ValueError("Gradient trace batch differs from the frozen policy")
        rate = config["learning_rate"]
        if config["warmup_steps"]:
            rate *= min(1, (row["step"] - 1) / config["warmup_steps"])
        if row["learning_rate"] != rate or not math.isfinite(row["gradient_l2"]) or row["gradient_l2"] < 0:
            raise ValueError("Gradient trace has an invalid norm or learning rate")
        if row["step"] in sampled and row["gradient_l2"] != sampled[row["step"]]:
            raise ValueError("Gradient trace disagrees with tensor diagnostics")
