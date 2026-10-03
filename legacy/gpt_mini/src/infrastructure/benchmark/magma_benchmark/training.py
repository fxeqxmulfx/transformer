"""Reuse the exact frozen training loop with an explicit factory extension.

The factory is replaced only during one synchronous run, then restored.
No measured source file is modified. Close-time instrumentation records
wrapper diagnostics and saves dense optimizer/RNG state in seed-0 checkpoints.
"""

import hashlib
from pathlib import Path
import torch

from gpt_mini.infrastructure.benchmark.optimizer_benchmark import runner as engine
from .registry import make_optimizer


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_tree(item) for item in value)
    return value


def train_one(config, data, attention, method, rate, seed, phase, steps, batch, *, checkpoints):
    captured = {}
    original_factory = engine.make_optimizer

    def factory(name, model, lr):
        model.magma_seed = seed
        optimizer = make_optimizer(name, model, lr)
        original_close = optimizer.close

        def close():
            captured["final_sha256"] = engine.state_hash(model)
            captured["magma"] = optimizer.diagnostics() if hasattr(optimizer, "diagnostics") else None
            if phase == "final" and seed == 0:
                path = checkpoints / f"{attention}_{method}_seed0.pt"
                if path.exists():
                    payload = torch.load(path, map_location="cpu", weights_only=True)
                    payload.update(optimizer=cpu_tree(optimizer.state_dict()), optimizer_steps=optimizer.steps,
                                   magma=captured["magma"], final_sha256=captured["final_sha256"])
                    torch.save(payload, path)
                    captured["checkpoint_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            original_close()

        optimizer.close = close
        return optimizer

    engine.make_optimizer = factory
    try:
        row = engine.train_one(config, data, attention, method, rate, seed, phase, steps, batch,
                               checkpoints=checkpoints)
    finally:
        engine.make_optimizer = original_factory
    row.update(captured)
    return row
