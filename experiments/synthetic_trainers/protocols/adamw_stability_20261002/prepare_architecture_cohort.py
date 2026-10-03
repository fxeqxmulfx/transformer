"""Execute two actual negative CPU six-pair cohorts, preserving native schedules."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import torch

from experiments.synthetic_trainers.architecture_cohort_protocol import freeze, run_cohort
from experiments.synthetic_trainers.architecture_cohort_report import assemble
from experiments.synthetic_trainers.architecture_training import ArchitectureRunConfig, make_model
from experiments.synthetic_trainers.attention_layout import digest
from experiments.synthetic_trainers.stability_report import write_json

PROTOCOL = Path(__file__).resolve().parent


def audit(stage, plan):
    result = {}
    for recipe in plan["recipes"]:
        checkpoint = stage / "cases" / recipe["name"] / recipe["name"] / "checkpoint.pt"
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        config = ArchitectureRunConfig(**recipe["config"])
        model = make_model(config, recipe["corpus"]["vocab_size"])
        model.load_state_dict(state["model"], strict=True)
        parameters = list(model.parameters())
        optimizer = torch.optim.AdamW(parameters, lr=config.learning_rate, betas=(.9, .98), eps=1e-8, weight_decay=.1)
        optimizer.load_state_dict(state["optimizer"])
        assert state["step"] == config.steps and len(optimizer.state) == 11
        assert sum(p.numel() for p in parameters) == 412296
        assert len(optimizer.param_groups) == 1
        assert {id(p) for p in optimizer.param_groups[0]["params"]} == {id(p) for p in parameters}
        for p in parameters:
            moments = optimizer.state[p]
            assert set(moments) == {"step", "exp_avg", "exp_avg_sq"}
            assert moments["step"].item() == config.steps and torch.isfinite(p).all()
            assert moments["exp_avg"].shape == moments["exp_avg_sq"].shape == p.shape
            assert all(torch.isfinite(v).all() and v.device.type == "cpu" for v in moments.values())
        group = optimizer.param_groups[0]
        assert group["betas"] == (.9, .98) and group["eps"] == 1e-8 and group["weight_decay"] == .1 and not group["amsgrad"]
        result[recipe["name"]] = {"checkpoint_sha256": digest(checkpoint), "completed_updates": config.steps,
            "parameters": 412296, "native_states": 11, "native_model_and_moments_finite": True,
            "maximum_buffer_absent": True, "betas": list(group["betas"]), "epsilon": group["eps"],
            "weight_decay": group["weight_decay"], "final_actual_rate": group["lr"],
            "audit_optimizer_updates_performed": 0}
    assert not torch.cuda.is_initialized()
    return result


def prepare(stage, archive):
    stage, archive = Path(stage), Path(archive)
    if stage.exists() or archive.exists():
        raise FileExistsError("Architecture cohort preparation requires fresh raw and archive paths")
    torch.set_num_threads(1); stage.mkdir(parents=True); archive.mkdir(parents=True)
    outcomes, audits = {}, {}
    for mode in ("normalizer", "schedule"):
        plan = freeze(stage / mode, PROTOCOL / "tagged-confirmation-preparation" / mode, CPU_fixture=True, render=False)
        assert run_cohort(stage / mode, plan)["completed_runs"] == 12
        outcomes[mode] = assemble(stage / mode, archive / mode, render=True)
        audits[mode] = audit(stage / mode, plan)
        assert outcomes[mode]["eligible_pair_support"] == 0
        assert not outcomes[mode]["repeatable_architecture_improvement"]
        print(json.dumps({"mode": mode, "completed_pairs": 6, "completed_runs": 12, "eligible_pair_support": 0}), flush=True)
    write_json(archive / "native-checkpoint-audits.json", audits)
    shutil.copyfile(__file__, archive / "preparation-snapshot.py")
    write_json(archive / "validation.json", {"recorded_utc": datetime.now(timezone.utc).isoformat(),
        "CPU_fixture": True, "scientific_run": False, "scientific_architecture_selected": False,
        "complete_CPU_cohorts": 2, "complete_CPU_pairs": 12, "complete_CPU_runs": 24,
        "total_CPU_updates": sum(r["total_updates"] for r in outcomes.values()),
        "all_native_checkpoints_independently_loaded_and_finite": True,
        "audit_optimizer_updates_performed": 0, "CUDA_initialized": False,
        "scope": "real_CPU_pipeline_preparation; no_scientific_learning_architecture_or_benchmark_claim"})
    write_json(archive / "artifact-hashes.json", {"files": {str(p.relative_to(archive)): digest(p)
        for p in sorted(archive.rglob("*")) if p.is_file() and p != archive / "artifact-hashes.json"}})
    return outcomes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    results = prepare(args.stage, args.archive)
    print(json.dumps({"complete_CPU_pairs": 12, "total_CPU_updates": sum(r["total_updates"] for r in results.values())}))
