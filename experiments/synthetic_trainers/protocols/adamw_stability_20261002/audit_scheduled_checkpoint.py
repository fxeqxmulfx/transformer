"""Inspect a completed native scheduled checkpoint on CPU, without updates."""

import argparse
from dataclasses import replace
import json
from pathlib import Path

import torch

from experiments.synthetic_trainers.paper_reproduction.grokking import make_model
from experiments.synthetic_trainers.scheduled_layout import current_sources, digest, validate_pair_plan
from experiments.synthetic_trainers.scheduled_rates import expected_rate
from experiments.synthetic_trainers.scheduled_training import ScheduledRunConfig


def audit(stage, name, *, archive=None):
    """Check the actual model, native moments and last applied scheduled rate."""
    if torch.cuda.is_initialized():
        raise ValueError("Checkpoint inspection requires an uninitialized CUDA context")
    stage = Path(stage)
    plan = validate_pair_plan(json.loads((stage / "plan.json").read_text()))
    if current_sources() != plan["source_hashes"]:
        raise ValueError("Frozen schedule sources changed")
    recipe = next((row for row in plan["recipes"] if row["name"] == name), None)
    if recipe is None:
        raise ValueError("Checkpoint case is absent from the frozen pair")
    case = stage / name
    if not (case / "measurements.json").exists():
        raise ValueError("A checkpoint audit requires the complete frozen case")
    report = json.loads((case / "measurements.json").read_text())
    config = ScheduledRunConfig(**recipe["config"])
    if (report["completed_steps"] != config.steps
            or report["plan"]["config"] != recipe["config"]
            or report["plan"]["source_hashes"] != plan["training_source_hashes"]):
        raise ValueError("A checkpoint audit requires the complete unchanged frozen case")
    checkpoint_path = case / "checkpoint.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint["step"] != config.steps:
        raise ValueError("Checkpoint update count differs from the full budget")
    if any(not torch.isfinite(value).all() for value in checkpoint["model"].values()):
        raise ValueError("Final model state contains a nonfinite tensor")
    torch.set_num_threads(1)
    model = make_model(replace(config, device="cpu"), plan["corpus"]["vocab_size"])
    model.load_state_dict(checkpoint["model"], strict=True)
    parameters = list(model.parameters())
    count = sum(parameter.numel() for parameter in parameters)
    if (count != report["plan"]["parameters"] or any(not parameter.requires_grad
            or not torch.isfinite(parameter).all() for parameter in parameters)):
        raise ValueError("Final model parameters differ or are nonfinite")
    optimizer = torch.optim.AdamW(parameters, lr=config.learning_rate, betas=(.9, .98),
                                 eps=1e-8, weight_decay=config.weight_decay)
    optimizer.load_state_dict(checkpoint["optimizer"])
    if type(optimizer) is not torch.optim.AdamW or len(optimizer.param_groups) != 1:
        raise ValueError("The actual optimizer must be one native AdamW group")
    group = optimizer.param_groups[0]
    rate = expected_rate(recipe["config"], config.steps - 1)
    if group["lr"] != rate:
        raise ValueError("Final native learning rate differs from the frozen schedule")
    if (group["betas"] != (.9, .98) or group["eps"] != 1e-8
            or group["weight_decay"] != config.weight_decay or group["amsgrad"]
            or group["maximize"] or len(group["params"]) != len(parameters)
            or {id(p) for p in group["params"]} != {id(p) for p in parameters}
            or len(optimizer.state) != len(parameters)):
        raise ValueError("Native settings or parameter coverage differ from the protocol")
    for parameter in parameters:
        state = optimizer.state[parameter]
        if set(state) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ValueError("Native moment keys differ from plain AdamW")
        if state["step"].item() != config.steps:
            raise ValueError("Native optimizer steps differ from the completed budget")
        if (state["exp_avg"].shape != parameter.shape
                or state["exp_avg_sq"].shape != parameter.shape
                or any(value.device.type != "cpu" or not torch.isfinite(value).all()
                       for value in state.values())):
            raise ValueError("Native moments have wrong shapes or nonfinite values")
    checkpoint_hash = digest(checkpoint_path)
    if archive is not None:
        manifest = json.loads((Path(archive) / "artifact-hashes.json").read_text())
        if manifest["raw_checkpoint_sha256"] != checkpoint_hash:
            raise ValueError("Actual checkpoint differs from the complete archive fingerprint")
    assert not torch.cuda.is_initialized()
    return {"name": name, "scientific_run": plan["scientific_run"],
        "completed_updates": config.steps, "parameters": count,
        "native_state_count": len(parameters), "all_native_steps": [config.steps],
        "actual_native_class": "torch.optim.AdamW", "betas": list(group["betas"]),
        "epsilon": group["eps"], "weight_decay": group["weight_decay"],
        "learning_rate_schedule": config.learning_rate_schedule, "learning_rate": group["lr"],
        "all_model_and_native_tensors_finite": True,
        "all_trainable_parameters_in_one_group_once": True,
        "maximum_second_moment_buffers_absent": True,
        "optimizer_updates_performed": 0, "GPU_context_initialized": False,
        "checkpoint_sha256": checkpoint_hash, "executed_audit_script_sha256": digest(__file__),
        "scope": "finite_completed_checkpoint_state_only; no_learning_or_stability_claim"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--case", choices=("adamw-constant", "adamw-cosine-tail"), required=True)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Checkpoint audit needs a fresh output; preserve previous receipts")
    result = audit(args.stage, args.case, archive=args.archive)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
