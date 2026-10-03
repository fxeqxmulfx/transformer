"""Opt-in native AdamW for existing complementary tasks and transfer metrics.

The unchanged common trainer owns sampling, loss, validation-only selection,
final and selected checkpoint tests, and separate novel-ID/length support.
Native optimizer hooks apply and record the explicit primary rate convention.
This adapter does not open the scientific benchmark or architecture gates.
"""

from dataclasses import asdict
import json
from pathlib import Path
from unittest.mock import patch

import torch

from . import training
from .complementary_config import ComplementaryConfig, learning_rate, optimizer_description
from .runtime import write_json


def cpu_state(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_state(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu_state(item) for item in value)
    return value


def optimizer_audit(optimizer, completed):
    """Describe actual groups and native state, including the zero-rate update."""
    if type(optimizer) is not torch.optim.AdamW:
        raise ValueError("Complementary optimization must use the native AdamW class")
    parameters = [p for group in optimizer.param_groups for p in group["params"]]
    states = [optimizer.state[p] for p in parameters]
    if (len({id(p) for p in parameters}) != len(parameters)
            or any(set(state) != {"step", "exp_avg", "exp_avg_sq"} for state in states)
            or any(int(state["step"].item()) != completed for state in states)
            or any(not torch.isfinite(value).all().item()
                   for state in states for value in state.values())):
        raise ValueError("Native AdamW state is incomplete, duplicated or nonfinite")
    return {"native_class": f"{type(optimizer).__module__}.{type(optimizer).__qualname__}",
            "completed_updates": completed, "parameter_tensors": len(parameters),
            "parameter_scalars": sum(p.numel() for p in parameters),
            "state_steps": [int(state["step"].item()) for state in states],
            "state_keys": [sorted(state) for state in states],
            "all_state_tensors_finite": True,
            "groups": [{key: group[key] for key in ("lr", "weight_decay", "betas", "eps", "amsgrad")}
                       | {"parameter_tensors": len(group["params"]),
                          "parameter_scalars": sum(p.numel() for p in group["params"])}
                       for group in optimizer.param_groups]}


def train(config, spec, model_spec, directory, *, model_factory=None, progress=None, prepared_splits=None):
    """Run a fresh full-budget study; never modify the modular frozen sources.

    This is an explicit protocol adaptation: complementary serialization and
    masked loss differ from modular division. No speed or transfer claim follows
    from sharing an optimizer. Existing generic defaults remain unchanged.
    """
    if type(config) is not ComplementaryConfig:
        raise TypeError("Complementary training requires an explicit ComplementaryConfig")
    if model_spec.init_std != .02:
        raise ValueError("The explicit primary matrix initialization is Normal(std=0.02)")
    if spec.task not in ("mqar", "lookup", "copy", "parity", "crasp"):
        raise ValueError("This adaptation covers MQAR, lookup/composition, copy/parity and prefix counting")
    directory = Path(directory)
    created = []
    completed = 0

    def optimizer_for(model, actual):
        if actual != config:
            raise ValueError("Common trainer changed the frozen complementary configuration")
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                     lr=learning_rate(config, 0), betas=(config.beta1, config.beta2),
                                     eps=config.optimizer_epsilon, weight_decay=config.weight_decay)
        created.append(optimizer)

        def before_step(native, args, kwargs):
            rate = learning_rate(config, completed)
            for group in native.param_groups:
                group["lr"] = rate

        def after_step(native, args, kwargs):
            nonlocal completed
            completed += 1
            record = {"step": completed, "learning_rates": [group["lr"] for group in native.param_groups],
                      "weight_decays": [group["weight_decay"] for group in native.param_groups]}
            with (directory / "update-rates.jsonl").open("a") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")

        optimizer.register_step_pre_hook(before_step)
        optimizer.register_step_post_hook(after_step)
        return optimizer

    with patch.object(training, "optimizer_for", optimizer_for), \
            patch.object(training, "optimizer_description", optimizer_description):
        result = training.train_run(spec, model_spec, config, directory, model_factory=model_factory,
                                    progress=progress, prepared_splits=prepared_splits)
    if len(created) != 1 or completed != config.steps or result["steps_completed"] != completed:
        raise ValueError("Complementary execution did not retain one full-budget native optimizer")
    audit = optimizer_audit(created[0], completed)
    torch.save(cpu_state(created[0].state_dict()), directory / "optimizer-final.pt")
    write_json(directory / "optimizer-audit.json", audit)
    result["complementary_protocol"] = {"config": asdict(config), "optimizer_audit": audit,
        "scope": "explicit_complementary_task_adaptation; scientific_gate_is_external"}
    write_json(directory / "result.json", result)
    return result
