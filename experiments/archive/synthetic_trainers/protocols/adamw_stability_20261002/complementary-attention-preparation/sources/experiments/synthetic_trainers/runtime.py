"""Timing, optimizer groups, and report provenance."""

import hashlib
import inspect
import json
from pathlib import Path

import torch


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def optimizer_for(model, config):
    if config.optimizer == "amsgradw":
        from experiments.optimizer_benchmark.coordinate import CoordinateOptimizer

        return CoordinateOptimizer(((name, p) for name, p in model.named_parameters() if p.requires_grad),
                                   config.learning_rate, rule="amsgrad", decay=config.weight_decay,
                                   beta=config.beta1, beta2=config.beta2, eps=config.optimizer_epsilon)
    parameters = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW([
        {"params": [p for p in parameters if p.ndim >= 2], "weight_decay": config.weight_decay},
        {"params": [p for p in parameters if p.ndim < 2], "weight_decay": 0.0},
    ], lr=config.learning_rate, betas=(config.beta1, config.beta2), eps=config.optimizer_epsilon)


def optimizer_description(config):
    return {"name": config.optimizer, "betas": [config.beta1, config.beta2],
            "epsilon": config.optimizer_epsilon, "bias_correction": config.optimizer == "adamw",
            "maximum_second_moment": config.optimizer == "amsgradw",
            "decoupled_decay": True, "decay_scope": "all_trainable_parameters" if config.optimizer == "amsgradw" else "matrices",
            "gradient_clip": config.grad_clip}


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def source_hashes(factory, optimizer=None):
    directory = Path(__file__).parent
    files = list(directory.glob("*.py")) + [directory.parent / "gpt_mini.py"]
    source = inspect.getsourcefile(factory)
    if source:
        files.append(Path(source))
    if optimizer is not None:
        source = inspect.getsourcefile(type(optimizer))
        if source:
            files.append(Path(source))
            if type(optimizer).__module__.startswith("experiments.optimizer_benchmark."):
                files.extend(Path(source).parent.glob("*.py"))
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(set(files)) if path.is_file()}
