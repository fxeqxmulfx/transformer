"""Freeze settings/hashes after profiling but before any learning run."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import torch

from lab.domain.spec import describe
from lab.infrastructure.benchmarks import build_task
from lab.infrastructure.loader import load
from lab.infrastructure.nn import build_model
from lab.infrastructure.provenance import code
from phase import write, require_budget

HERE = Path(__file__).resolve().parent


def main():
    path = HERE / "protocol.json"
    if path.exists():
        raise FileExistsError("The learning declaration is already frozen")
    torch.set_num_threads(1)
    study = load(HERE)
    primary_allowance = min(360, max(60, int((require_budget() - 1200) / 14)))
    cases = []
    for label, experiment in study.experiments.items():
        task = build_task(experiment.benchmark, experiment.seeds.data, torch.device("cpu"))
        model = build_model(experiment.model, task.vocab, experiment.seeds.model)
        parameters = torch.cat([p.detach().flatten() for p in model.parameters()])
        common = torch.cat([p.detach().flatten() for name, p in model.named_parameters()
                            if not name.startswith("residual_mixers")])
        cases.append({"label": label, "experiment": describe(experiment),
            "learning_wall_seconds": primary_allowance if label.startswith("division") else primary_allowance / 2,
            "parameters": parameters.numel(),
            "initial_parameter_sha256": hashlib.sha256(parameters.numpy().astype("<f4").tobytes()).hexdigest(),
            "common_parameter_sha256": hashlib.sha256(common.numpy().astype("<f4").tobytes()).hexdigest(),
            "task": task.metadata})
    sources = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob("*.py")}
    write(path, {"frozen_utc": datetime.now(timezone.utc).isoformat(), "cases": cases,
        "engine_sources": code(), "cohort_sources": sources,
        "criterion": "train and validation >=0.999 for 3 consecutive canonical observations, all >=5000 updates",
        "selection": "current endpoint; no best restoration; final test only after learning_closed",
        "final_scoring": "all final train/validation/test scores replayed on CPU FP32; report CUDA-validation drift; FP64 audit fixes all dtype RMS epsilons at FP32 values",
        "optimizer_backend": "foreach revised Schedule-Free AdamW; same recurrence as lean_optim",
        "attention_backend": "explicit mathematical SDPA, fullgraph max-autotune",
        "maximum_gpu_phase_seconds": 7200, "global_final_evaluation_reserve_seconds": 180,
        "allocation": "fixed order; before-learning primary allowance=min(360,floor((remaining_phase-1200)/14)) with floor60s; mechanisms half; compilation charged globally; check time each1000 updates; one fixed-order continuation round shares remaining time equally among remaining censored cases, less30s compilation allowance per case; no failed-run retries",
        "interpretation": "compare curves at common updates and wall time; unfinished means censored, not impossible",
        "hypotheses": [
            "conditioned Q/K may outperform raw Q/K if stored query/base/average RMS grows beyond2",
            "if coordinate scales remain1, a large-query conditioning mechanism did not activate",
            "key-only and branch-scale controls test different hypotheses and parameterizations",
            "full-transformer learning/generalization is not implied by the proved coordinate identities"]})


if __name__ == "__main__":
    main()
