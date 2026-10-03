"""Inspect an already completed native normalizer checkpoint on CPU only."""
import json
from pathlib import Path
import sys
from dataclasses import replace
import torch

from experiments.synthetic_trainers.attention_layout import digest
from experiments.synthetic_trainers.attention_training import AttentionRunConfig, make_model

name = sys.argv[1]
assert name in ("adamw-sparsemax", "adamw-softmax")
torch.set_num_threads(1)
protocol = Path("experiments/synthetic_trainers/protocols/adamw_stability_20261002")
plan = json.loads((protocol / "attention-pair-plan.json").read_text())
config = AttentionRunConfig(**next(r for r in plan["recipes"] if r["name"] == name)["config"])
case = Path(plan["output_directory"]) / name
report = json.loads((case / "measurements.json").read_text())
assert report["completed_steps"] == 300000 and report["plan"]["config"] == config.__dict__
checkpoint = torch.load(case / "checkpoint.pt", map_location="cpu", weights_only=True)
assert checkpoint["step"] == 300000
model = make_model(replace(config, device="cpu"), plan["corpus"]["vocab_size"])
model.load_state_dict(checkpoint["model"], strict=True)
parameters = list(model.parameters())
assert sum(p.numel() for p in parameters) == 436104
assert len(parameters) == 11 and all(p.requires_grad and torch.isfinite(p).all() for p in parameters)
optimizer = torch.optim.AdamW(parameters, lr=config.learning_rate, betas=(.9, .98),
                             eps=1e-8, weight_decay=config.weight_decay)
optimizer.load_state_dict(checkpoint["optimizer"])
assert type(optimizer) is torch.optim.AdamW and len(optimizer.param_groups) == 1
group = optimizer.param_groups[0]
assert group["betas"] == (.9, .98) and group["eps"] == 1e-8
assert group["lr"] == config.learning_rate and group["weight_decay"] == config.weight_decay
assert not group["amsgrad"] and not group["maximize"]
assert {id(p) for p in group["params"]} == {id(p) for p in parameters}
assert len(optimizer.state) == len(parameters)
for p in parameters:
    state = optimizer.state[p]
    assert set(state) == {"step", "exp_avg", "exp_avg_sq"}
    assert state["step"].item() == 300000
    assert state["exp_avg"].shape == state["exp_avg_sq"].shape == p.shape
    assert all(torch.isfinite(v).all() and v.device.type == "cpu" for v in state.values())
assert not torch.cuda.is_initialized()
archive = Path("experiments/synthetic_trainers/baselines") / (
    f"adamw_attention_{name}_mod193_fraction25_lr0003_budget300k_seed0_data0_20261002")
checkpoint_hash = digest(case / "checkpoint.pt")
if archive.exists():
    assert json.loads((archive / "artifact-hashes.json").read_text())["raw_checkpoint_sha256"] == checkpoint_hash
result = {"name": name, "completed_updates": 300000, "parameters": 436104,
          "native_state_count": 11, "all_native_steps": [300000],
          "all_model_and_native_tensors_finite": True,
          "maximum_second_moment_buffers_absent": True,
          "actual_native_class": "torch.optim.AdamW", "betas": list(group["betas"]),
          "epsilon": group["eps"], "weight_decay": group["weight_decay"],
          "amsgrad": group["amsgrad"], "learning_rate": group["lr"],
          "all_trainable_parameters_in_one_group_once": True,
          "optimizer_updates_performed": 0, "GPU_context_initialized": False,
          "checkpoint_sha256": checkpoint_hash, "executed_audit_script_sha256": digest(__file__)}
output = case.parent.parent / "bootstrap" / f"attention-{name.removeprefix('adamw-')}-final-CPU-audit.json"
assert not output.exists()
output.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
