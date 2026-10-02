"""Check scheduled logs with every original non-rate integrity requirement.

Adapted from this repository's stability_integrity.py: only the expected rate
calculation changes. Stored rates and every other recorded value remain intact.
"""

import math

from .scheduled_rates import expected_rate, validate_schedule
from .stability_integrity import expected_batches, validate_observation


def validate_gradient_trace(report, diagnostics, gradients, manifest):
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
        rate = expected_rate(config, row["step"] - 1)
        if row["learning_rate"] != rate or not math.isfinite(row["gradient_l2"]) or row["gradient_l2"] < 0:
            raise ValueError("Gradient trace has an invalid norm or scheduled learning rate")
        if row["step"] in sampled and row["gradient_l2"] != sampled[row["step"]]:
            raise ValueError("Gradient trace disagrees with tensor diagnostics")


def validate_logs(report, diagnostics, probes, manifest, gradients=()):
    config, instrumentation = report["plan"]["config"], manifest["instrumentation"]
    validate_schedule(config)
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
    required_norms = {"parameter_l2", "parameter_before_l2", "gradient_l2", "update_l2",
                      "exp_avg_l2", "exp_avg_sq_l2"}
    for row in diagnostics:
        expected = expected_batches(config, row["step"])
        for key in ("batch_size", "cursor_before", "cursor_after", "epoch_tail", "epoch_wraps_in_batch"):
            if row[key] != expected[key]:
                raise ValueError(f"Diagnostic {key} differs from the declared batch policy")
        if set(row["parameters"]) != parameter_names:
            raise ValueError("Diagnostic parameter support changed")
        if not parameter_names or any(not required_norms <= norms.keys() for norms in row["parameters"].values()):
            raise ValueError("Diagnostic optimizer moment or parameter norms are missing")
        if not math.isclose(row["learning_rate"], expected_rate(config, row["step"] - 1), rel_tol=1e-12, abs_tol=1e-15):
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
