"""Measure the original synthetic and memorization trainers at their fixture commits."""

from pathlib import Path
import tempfile

import torch

from common import hashes, records


def without_generation_costs(value):
    if isinstance(value, dict):
        return {key: without_generation_costs(item) for key, item in value.items()
                if key not in ("seconds", "forward_calls", "processed_tokens", "tokens_per_second",
                               "generation_forward_calls", "generation_padded_tokens_processed",
                               "generation_attention_cells_processed", "generation_seconds")}
    if isinstance(value, list):
        return [without_generation_costs(item) for item in value]
    return value


def replay(golden):
    from experiments.synthetic_trainers.config import ModelSpec, TrainConfig
    from experiments.synthetic_trainers.specs import TaskSpec
    from experiments.synthetic_trainers.training import train_run

    for expected in golden["runs"].values():
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            result = train_run(TaskSpec(**expected["task"]), ModelSpec(**expected["model"]),
                               TrainConfig(**expected["training"]), directory)
            history = records(directory / "history.jsonl")
            if "study" not in expected:
                history = [{key: row[key] for key in ("step", "examples_seen", "validation")}
                           for row in history]
            expected.update(history=without_generation_costs(history))
            for key in ("best_step", "validation_best", "test", "time_to_target", "split_fingerprints"):
                expected[key] = without_generation_costs(result[key])
            if "study" in expected:
                expected.update(study=without_generation_costs(result["study"]),
                                test_final=without_generation_costs(result["test_final"]))
            expected["parameters"] = {name: hashes(torch.load(directory / f"{name}.pt", weights_only=True))
                                      for name in ("best", "final")}
