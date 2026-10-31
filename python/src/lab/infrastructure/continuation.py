"""Verified full-state Schedule-Free continuation; source files are read-only.

Source: domain.continuation.ScheduleFreeContinuation. The actual optimizer and
training loop are reused. This initialization changes only configured rate
and decay, retaining all histories and the exact next sampler state.
"""

import hashlib
import json
from pathlib import Path

import torch

from ..domain.spec import describe


def source_checkpoint(experiment, device):
    policy = experiment.checkpoint
    path = Path(policy.source)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[4] / path
    if hashlib.sha256(path.read_bytes()).hexdigest() != policy.sha256:
        raise ValueError("Continuation checkpoint SHA256 mismatch")
    directory = path.parent.parent if path.parent.name == "checkpoints" else path.parent
    original = json.loads((directory / "experiment.json").read_text())["description"]
    current = describe(experiment)
    for key in ("model", "benchmark", "seeds"):
        if original[key] != current[key]:
            raise ValueError(f"Continuation {key} differs from the source")
    if original["budget"]["batch"] != current["budget"]["batch"]:
        raise ValueError("Continuation batch size differs from the source")
    state = torch.load(path, map_location=device, weights_only=True)
    groups = state["optimizer"]["param_groups"]
    if len(groups) != 1 or groups[0].get("training") is not True or "maximum" not in groups[0]:
        raise ValueError("Continuation requires a revised Schedule-Free training-query state")
    group = groups[0]
    for name in ("betas", "eps", "warmup"):
        if group[name] != getattr(experiment.optimizer, name):
            raise ValueError(f"Continuation {name} differs from the source")
    if group["step"] != state["step"] or state["step"] >= experiment.budget.updates:
        raise ValueError("Continuation needs an aligned optimizer clock before its absolute budget")
    group["lr"] = experiment.optimizer.lr
    group["weight_decay"] = experiment.optimizer.weight_decay
    return state
