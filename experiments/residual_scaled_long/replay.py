"""Replay task154 endpoints and compare matched first-99% observations.

Reuse task145's independent CPU scorer, including its below-epsilon XSA
convention and fixed-FP32-epsilon FP64 audit. The previously published test
split is reused for endpoint reporting, never for continuation or stopping.
"""

import copy
import importlib.util
import json
import math
import platform
import sys
import time

import torch

from lab.infrastructure.benchmarks import build_task
from lab.infrastructure.loader import load
from lab.infrastructure.nn import build_model
from lab.infrastructure.optim.residual_schedule_free import ResidualScheduleFreeAdamW
from lab.infrastructure.store import RunDirectory

from cohort import HERE, PARENT, PHASE, PROTOCOL, TARGET, sha, write

# Import only the existing scorer; do not execute its closed-cohort main().
sys.path.insert(0, str(PARENT))
module_spec = importlib.util.spec_from_file_location("parent_analysis", PARENT / "analyze.py")
parent_analysis = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(parent_analysis)


def replay():
    phase = json.loads(PHASE.read_text())
    if phase["status"] != "learning_closed":
        raise ValueError("Both learning runs must close before final test scoring")
    protocol = json.loads(PROTOCOL.read_text())
    if sha(PARENT / "analyze.py") != protocol["parent_scoring_sha256"]:
        raise ValueError("Frozen parent scorer changed")
    parent_results = json.loads((PARENT / "results.json").read_text())
    torch.set_num_threads(1)
    started = time.monotonic()
    cases = []
    study = load(HERE)
    for declaration in protocol["cases"]:
        label = declaration["label"]
        spec = study.experiments[label]
        run = RunDirectory(HERE / "runs" / label)
        checkpoint_hash = sha(run.path / "checkpoint.pt")
        checkpoint = run.checkpoint("cpu")
        task = build_task(spec.benchmark, spec.seeds.data, torch.device("cpu"))
        if task.metadata != declaration["task"]:
            raise ValueError("Frozen task fingerprints changed")
        model = build_model(spec.model, task.vocab, spec.seeds.model)
        optimizer = ResidualScheduleFreeAdamW(model.parameters(), **vars(spec.optimizer))
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        optimizer.eval()
        model.eval()
        precise = copy.deepcopy(model)
        for module in precise.modules():
            if isinstance(module, torch.nn.RMSNorm) and module.eps is None:
                module.eps = torch.finfo(torch.float32).eps
        precise.double()
        metrics, audits = {}, {}
        for name in ("train", "validation", "test"):
            metrics[name], prediction = parent_analysis.score(model, task.splits[name])
            audits[name], reference = parent_analysis.score(precise, task.splits[name])
            audits[name]["fp32_prediction_disagreements"] = int((prediction != reference).sum())
        history = run.records("history")
        gradients = run.records("gradients")
        if [row["step"] for row in gradients] != list(range(1, checkpoint["step"] + 1)):
            raise ValueError("Incomplete gradient trace")
        if any(not math.isfinite(row["gradient_l2"]) for row in gradients):
            raise FloatingPointError("Nonfinite gradient")
        crossing = next((row for row in history if row["validation"]["accuracy"] >= TARGET), None)
        if crossing is None or crossing["step"] != checkpoint["step"]:
            raise AssertionError("Endpoint is not the first canonical 99% crossing")
        if metrics["validation"]["accuracy"] < TARGET:
            raise AssertionError("CPU replay did not confirm the requested validation target")
        ordinary = next(case for case in parent_results["cases"]
                        if case["mode"] == "add" and case["prime"] == 97 and case["seed"] == spec.seeds.model)
        ordinary_crossing = next(row for row in ordinary["learning_curve"] if row["validation"]["accuracy"] >= TARGET)
        parent = next(case for case in parent_results["cases"] if case["label"] == declaration["parent_label"])
        case = {"label": label, "parent_label": declaration["parent_label"], "mode": "scaled",
            "prime": 97, "seed": spec.seeds.model, "step": checkpoint["step"], "stop": run.result()["stop"],
            "parameters": sum(p.numel() for p in model.parameters()),
            "checkpoint_sha256": checkpoint_hash, "parent_checkpoint_sha256": declaration["parent_files"]["checkpoint.pt"],
            "additional_updates": checkpoint["step"] - declaration["from_step"],
            "first_validation99_update": crossing["step"], "stable_validation999": spec.benchmark.stable(history),
            "final_cpu_fp32": metrics, "fp64_fixed_epsilon_audit": audits,
            "cpu_minus_cuda_correct": {name: metrics[name]["correct"] - round(crossing[name]["accuracy"] * metrics[name]["examples"]) for name in ("train", "validation")},
            "ordinary_first99": {"step": ordinary_crossing["step"], "training_seconds": ordinary_crossing["training_seconds"], "wall_seconds": ordinary_crossing["wall_seconds"]},
            "update_ratio_ordinary_over_scaled": ordinary_crossing["step"] / crossing["step"],
            "source_wall_seconds": checkpoint["wall_seconds"],
            "training_seconds": crossing["training_seconds"],
            "additional_training_seconds": crossing["training_seconds"] - parent["learning_curve"][-1]["training_seconds"],
            "additional_wall_seconds": checkpoint["wall_seconds"] - parent["source_wall_seconds"],
            "gradients": {"count": len(gradients), "all_finite": True, "sha256": sha(run.path / "gradients.jsonl")},
            "learning_curve": history}
        if sha(run.path / "checkpoint.pt") != checkpoint_hash:
            raise AssertionError("Replay changed the checkpoint")
        for name, digest in declaration["parent_files"].items():
            if sha(PARENT / "runs" / declaration["parent_label"] / name) != digest:
                raise AssertionError("Continuation changed the original cohort")
        cases.append(case)
        print(json.dumps({key: case[key] for key in ("label", "step", "additional_updates", "final_cpu_fp32", "ordinary_first99")}), flush=True)
    report = {"protocol_sha256": sha(PROTOCOL), "phase": phase, "cases": cases,
        "python": platform.python_version(), "torch": torch.__version__,
        "scope": protocol["scope"], "replay_seconds": time.monotonic() - started}
    write(HERE / "results.json", report)


if __name__ == "__main__":
    replay()
