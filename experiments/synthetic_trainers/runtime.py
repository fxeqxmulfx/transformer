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
    parameters = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW([
        {"params": [p for p in parameters if p.ndim >= 2], "weight_decay": config.weight_decay},
        {"params": [p for p in parameters if p.ndim < 2], "weight_decay": 0.0},
    ], lr=config.learning_rate)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def source_hashes(factory):
    directory = Path(__file__).parent
    files = list(directory.glob("*.py")) + [directory.parent / "gpt_mini.py"]
    source = inspect.getsourcefile(factory)
    if source:
        files.append(Path(source))
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(set(files)) if path.is_file()}
