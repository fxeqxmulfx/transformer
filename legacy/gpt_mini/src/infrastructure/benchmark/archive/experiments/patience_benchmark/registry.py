"""Combine all 24 frozen recipes; retain their validation-selected rates."""

import json
from pathlib import Path

from experiments.optimizer_benchmark.registry import METHODS as OLD_METHODS, make_optimizer as old_optimizer
from experiments.magma_benchmark.registry import METHODS as MAGMA_METHODS, make_optimizer as magma_optimizer


METHODS = OLD_METHODS + MAGMA_METHODS
PAIRED_NAMES = {"sgd", "adam", "adamw", "muon", "rmsprop",
                "magma_sgd", "magma_adam", "magma_adamw", "magma_muon", "magma_rmsprop"}


def selected_rates():
    old = json.loads(Path("experiments/optimizer_benchmark/results/rtx3050/selected_rates.json").read_text())
    new = json.loads(Path("experiments/magma_benchmark/results/rtx3050/selected_rates.json").read_text())
    return {attention: old[attention] | new[attention] for attention in ("softmax", "sparsemax")}


def make_optimizer(name, model, lr, seed):
    if name == "rmsprop" or name.startswith("magma_"):
        model.magma_seed = seed
        return magma_optimizer(name, model, lr)
    return old_optimizer(name, model, lr)
