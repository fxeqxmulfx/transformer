"""Replay every retained endpoint after learning closes; open the reserved test.

CPU FP32 is the predeclared final scorer. FP64 keeps dtype-dependent RMS eps
at the FP32 value, preserving the same finite-epsilon model. Neither replay
selects a checkpoint or starts another learning run. All incomplete cases,
CUDA/CPU count differences, curves and gradient traces remain visible.
"""

import copy
import hashlib
import json
import math
from pathlib import Path
import platform
import time

import torch
from torch.nn import functional as F

from lab.infrastructure.benchmarks import build_task
from lab.infrastructure.loader import load
from lab.infrastructure.nn import build_model
from lab.infrastructure.optim.residual_schedule_free import ResidualScheduleFreeAdamW
from lab.infrastructure.store import RunDirectory
from phase import LEDGER, write

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@torch.no_grad()
def score(model, rows, batch=512):
    total_loss, correct, predictions, margins = 0., 0, [], []
    for part in rows.split(batch):
        output = model(part[:, :-1])[:, -1]
        if not bool(torch.isfinite(output).all()):
            raise FloatingPointError("Nonfinite final logits")
        target = part[:, -1]
        total_loss += float(F.cross_entropy(output, target, reduction="sum"))
        prediction = output.argmax(-1)
        correct += int((prediction == target).sum())
        predictions.append(prediction)
        actual = output.gather(1, target[:, None]).squeeze(1)
        competing = output.clone().scatter_(1, target[:, None], -torch.inf).amax(1)
        margins.append(actual - competing)
    predictions, margins = torch.cat(predictions), torch.cat(margins)
    return {"examples": len(rows), "correct": correct, "accuracy": correct / len(rows),
        "loss": total_loss / len(rows), "minimum_target_margin": float(margins.min()),
        "prediction_sha256": hashlib.sha256(predictions.numpy().astype("<i8").tobytes()).hexdigest()}, predictions


def final_case(declaration, spec):
    label = declaration["label"]
    run = RunDirectory(HERE / "runs" / label)
    if not (run.path / "checkpoint.pt").exists():
        return {"label": label, "status": "no_checkpoint"}
    checkpoint_hash = sha(run.path / "checkpoint.pt")
    checkpoint = run.checkpoint("cpu")
    task = build_task(spec.benchmark, spec.seeds.data, torch.device("cpu"))
    if task.metadata != declaration["task"]:
        raise ValueError("Frozen data fingerprints changed")
    model = build_model(spec.model, task.vocab, spec.seeds.model)
    optimizer = ResidualScheduleFreeAdamW(model.parameters(), **vars(spec.optimizer))
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    optimizer.eval()
    model.eval()
    history = [row for row in run.records("history") if row["step"] <= checkpoint["step"]]
    gradients = [row for row in run.records("gradients") if row["step"] <= checkpoint["step"]]
    if [row["step"] for row in gradients] != list(range(1, checkpoint["step"] + 1)):
        raise ValueError("Incomplete saved gradient trace")
    if any(not math.isfinite(row["gradient_l2"]) for row in gradients):
        raise FloatingPointError("Nonfinite gradient in retained trace")
    precise = copy.deepcopy(model)
    for module in precise.modules():
        if isinstance(module, torch.nn.RMSNorm) and module.eps is None:
            module.eps = torch.finfo(torch.float32).eps
    precise.double()
    metrics, audit = {}, {}
    for name in ("train", "validation", "test"):
        metrics[name], prediction = score(model, task.splits[name])
        audit[name], reference = score(precise, task.splits[name])
        audit[name]["fp32_prediction_disagreements"] = int((prediction != reference).sum())
    last = next((row for row in reversed(history) if row["step"] == checkpoint["step"]), None)
    cuda_drift = None if last is None else {
        name: metrics[name]["correct"] - round(last[name]["accuracy"] * metrics[name]["examples"])
        for name in ("train", "validation")}
    curve = [{k: row[k] for k in ("step", "train", "validation", "training_seconds", "wall_seconds", "depth_queries")}
             for row in history]
    first99 = next((row["step"] for row in history if row["validation"]["accuracy"] >= .99), None)
    valid = spec.benchmark.stable(history)
    result = run.result()
    original_stop = None if result is None else result["stop"]
    if original_stop and original_stop["reason"] == "stable_validation" and not valid:
        raise AssertionError("Claimed stable validation is absent from retained history")
    if sha(run.path / "checkpoint.pt") != checkpoint_hash:
        raise AssertionError("Replay changed the saved checkpoint")
    norms = [row["gradient_l2"] for row in gradients]
    return {"label": label, "mode": spec.model.mixing, "prime": spec.benchmark.prime,
        "seed": spec.seeds.model, "step": checkpoint["step"], "stop": original_stop,
        "stable_validation": valid, "first_validation99_update": first99,
        "checkpoint_sha256": checkpoint_hash, "parameters": declaration["parameters"],
        "source_wall_seconds": checkpoint["wall_seconds"], "final_cpu_fp32": metrics,
        "fp64_fixed_epsilon_audit": audit, "cpu_minus_cuda_correct": cuda_drift,
        "gradients": {"count": len(norms), "minimum": min(norms) if norms else None,
            "maximum": max(norms) if norms else None, "last": norms[-1] if norms else None,
            "trace_sha256": sha(run.path / "gradients.jsonl") if norms else None},
        "max_coordinate_scale": max((q["coordinate_scale"] for row in history for q in row["depth_queries"]), default=1.),
        "learning_curve": curve}


def main():
    phase = json.loads(LEDGER.read_text())
    if phase["status"] not in ("learning_closed", "closed"):
        raise ValueError("Final test remains sealed until the learning phase closes")
    torch.set_num_threads(1)
    start = time.monotonic()
    protocol = json.loads((HERE / "protocol.json").read_text())
    study = load(HERE)
    cases = []
    for declaration in protocol["cases"]:
        case = final_case(declaration, study.experiments[declaration["label"]])
        cases.append(case)
        print(json.dumps({k: v for k, v in case.items() if k not in ("learning_curve", "fp64_fixed_epsilon_audit")}), flush=True)
    report = {"protocol_sha256": sha(HERE / "protocol.json"), "phase": phase,
        "python": platform.python_version(), "torch": torch.__version__, "cases": cases,
        "scope": "disjoint operand pairs at each fixed prime; two initialization seeds; no integer/prime/length OOD claim"}
    write(HERE / "results.json", report)
    print(json.dumps({"replay_seconds": time.monotonic() - start, "cases": len(cases)}), flush=True)


if __name__ == "__main__":
    main()
