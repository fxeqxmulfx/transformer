"""Portable interpretation of explicitly tagged fixed learning-rate schedules."""

import math


def validate_schedule(config):
    if (config["model"] != "gptmini" or config["optimizer"] != "adamw"
            or config["learning_rate_schedule"] not in ("constant", "cosine_tail")
            or not 0 < config["steps"] <= 300000):
        raise ValueError("Schedule calibration requires native AdamW, softmax GPTMini and the user cap")
    if (type(config["anneal_start"]) is not int or type(config["anneal_end"]) is not int
            or not config["warmup_steps"] <= config["anneal_start"] < config["anneal_end"] <= config["steps"]):
        raise ValueError("Fixed annealing must follow warmup and fit inside the budget")
    if (not math.isfinite(config["final_rate_factor"]) or not 0 < config["final_rate_factor"] < 1
            or not math.isfinite(config["learning_rate"]) or config["learning_rate"] <= 0):
        raise ValueError("Initial and final rates must be finite, positive and strictly reduced")


def expected_rate(config, completed_steps):
    """Rate applied to the next update after exactly completed_steps updates."""
    if type(completed_steps) is not int or completed_steps < 0:
        raise TypeError("Completed update count must be a nonnegative integer")
    initial = config["learning_rate"]
    if config["warmup_steps"]:
        initial *= min(1, completed_steps / max(1, config["warmup_steps"]))
    if config["learning_rate_schedule"] == "constant" or completed_steps <= config["anneal_start"]:
        return initial
    fraction = min(1, (completed_steps - config["anneal_start"]) / (config["anneal_end"] - config["anneal_start"]))
    factor = config["final_rate_factor"] + (1 - config["final_rate_factor"]) * (1 + math.cos(math.pi * fraction)) / 2
    return initial * factor
