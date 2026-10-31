"""Replay actual updates and evaluate the prospectively declared transfer tests.

Source: hypotheses.json and review/flat_spectrum.md, learning transfer.
CPU float64 replays retain the trained model's float32 RMS epsilon. The next
batch and complete optimizer history come from each saved checkpoint. These
independent diagnostic copies never alter training or select hyperparameters.
From python/: uv run --locked python ../experiments/rotate90_transfer/analyze.py
"""

import csv
import hashlib
import json
import math
from pathlib import Path
import platform
from statistics import median

import torch
from torch.nn import functional as F

from lab.domain.training import rate
from lab.infrastructure.benchmarks import build_task
from lab.infrastructure.benchmarks.samplers import gather
from lab.infrastructure.loader import load
from lab.infrastructure.nn import build_model
from lab.infrastructure.optim import build_optimizer

ROOT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def double_model(spec):
    model = build_model(spec.model, spec.benchmark.prime, spec.seeds.model).double()
    model.normalize = lambda x: F.rms_norm(x, (x.size(-1),), eps=torch.finfo(torch.float32).eps)
    return model


def measure(model, rows, gradient=False):
    model.zero_grad(set_to_none=True)
    loss_sum, correct = 0., 0
    with torch.set_grad_enabled(gradient):
        for chunk in rows.split(1024):
            logits = model(chunk[:, :2])
            loss = F.cross_entropy(logits, chunk[:, 2], reduction="sum") / len(rows)
            if gradient:
                loss.backward()
            loss_sum += float(loss.detach())
            correct += int((logits.detach().argmax(-1) == chunk[:, 2]).sum())
    return {"loss": loss_sum, "accuracy": correct / len(rows)}


@torch.no_grad()
def displacement(optimizer, next_rate):
    """Derive all evaluation coordinates from the independently stored y,z,v."""
    group = optimizer.param_groups[0]
    assert group["training"]
    beta1, beta2 = group["betas"]
    step = group["step"] + 1
    eta = next_rate * (min(1., step / group["warmup"]) if group["warmup"] else 1.)
    maximum = max(eta, group["maximum"])
    total = group["total"] + maximum ** 2
    c = maximum ** 2 / total if total else 0.
    result = []
    for parameter in group["params"]:
        y, g, state = parameter.detach(), parameter.grad.detach(), optimizer.state[parameter]
        z = state["base"]
        x = y + (1 - 1 / beta1) * (z - y)
        second = beta2 * state["second"] + (1 - beta2) * g.square()
        denominator = (second / (1 - beta2 ** step)).sqrt() + group["eps"]
        averaging = c * (z - x)
        transfer = -c * eta * g / denominator
        decay = -c * eta * group["weight_decay"] * y
        result.append({"x": x, "averaging": averaging, "transfer": transfer, "decay": decay,
                       "delta": averaging + transfer + decay})
    return result, {"rate": eta, "coefficient": c, "next_update": step}


def probe(spec, checkpoint):
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = double_model(spec)
    model.load_state_dict(state["model"])
    optimizer = build_optimizer(spec.optimizer, model)
    optimizer.load_state_dict(state["optimizer"])
    assert optimizer.param_groups[0]["step"] == state["step"]
    task = build_task(spec.benchmark, spec.seeds.data, torch.device("cpu"))
    sampler = task.sampler(spec.budget.batch, spec.seeds.batch_seed)
    sampler.restore(state)
    parts, _ = sampler.next()
    batch = task.inputs(gather(parts))
    training = measure(model, batch, gradient=True)
    next_rate = rate(spec.optimizer.lr, spec.schedule, state["step"])
    pieces, scalars = displacement(optimizer, next_rate)
    evaluated = double_model(spec)
    with torch.no_grad():
        for parameter, piece in zip(evaluated.parameters(), pieces, strict=True):
            parameter.copy_(piece["x"])
    before = measure(evaluated, task.splits["heldout"], gradient=True)
    terms = {key: sum(float((p.grad * piece[key]).sum())
                     for p, piece in zip(evaluated.parameters(), pieces, strict=True))
             for key in ("averaging", "transfer", "decay", "delta")}
    norm_sq = sum(float(piece["delta"].square().sum()) for piece in pieces)
    optimizer.param_groups[0]["lr"] = next_rate
    optimizer.step()
    optimizer.eval()
    identity_error = max(float((p - piece["x"] - piece["delta"]).abs().max().detach())
                         for p, piece in zip(model.parameters(), pieces, strict=True))
    after = measure(model, task.splits["heldout"])
    change = after["loss"] - before["loss"]
    predicted = terms["delta"]
    eligible = abs(predicted) > 1e-8
    return {"step": state["step"], "checkpoint_sha256": digest(checkpoint),
            "next_batch_sha256": hashlib.sha256(batch.numpy().astype("<i8").tobytes()).hexdigest(),
            "batch_size": len(batch), "task": task.metadata, **scalars,
            "query_batch": training, "before": before, "after": after,
            "directional_terms": terms, "delta_norm_sq": norm_sq,
            "evaluation_identity_max_error": identity_error,
            "actual_change": change, "remainder": change - predicted,
            "remainder_over_norm_sq": (change - predicted) / norm_sq if norm_sq else None,
            "relative_first_order_progress": -predicted / before["loss"],
            "sign_eligible": eligible, "sign_correct": bool(eligible and predicted * change > 0)}


def main():
    torch.set_num_threads(1)
    hypotheses = json.loads((ROOT / "hypotheses.json").read_text())
    findings, curves = [], []
    for label, spec in load(ROOT).experiments.items():
        directory = ROOT / "runs" / label
        result = json.loads((directory / "result.json").read_text())
        manifest = json.loads((directory / "experiment.json").read_text())
        assert result["updates"] == 1500
        assert manifest["segments"][-1]["source_sha256"] == digest(ROOT / "experiment.py")
        probes = [probe(spec, directory / "checkpoints" / f"step-{step:08d}.pt")
                  for step in hypotheses["probe_steps"]]
        # probe.step is the saved step, not the update being replayed.
        by_step = dict(zip(hypotheses["probe_steps"], probes, strict=True))
        early = median(by_step[t]["relative_first_order_progress"] for t in (100, 200, 400))
        log_forecast = math.log(by_step[400]["before"]["loss"]) - early * 1100
        forecast = early > 0 and log_forecast <= math.log(.01 * math.log(2))
        success = result["final"]["heldout"]["accuracy"] >= .99
        record = {"label": label, "prime": spec.benchmark.prime, "exponent": spec.benchmark.exponent,
                  "width": spec.model.width, "seed": spec.seeds.model,
                  "train": result["final"]["train"], "heldout": result["final"]["heldout"],
                  "early_relative_progress": early, "log_forecast_loss": log_forecast,
                  "forecast_success99": forecast, "success99": success,
                  "false_positive": forecast and not success, "false_negative": success and not forecast,
                  "first_heldout99_update": result.get("first_heldout99_update"),
                  "training_seconds": result["training_seconds"],
                  "provenance": manifest["segments"][-1]["provenance"], "probes": probes}
        findings.append(record)
        for line in (directory / "history.jsonl").read_text().splitlines():
            row = json.loads(line)
            for split in ("train", "heldout"):
                curves.append({"label": label, "step": row["step"], "split": split,
                               "loss": row[split]["loss"], "accuracy": row[split]["accuracy"]})
        print(label, "accuracy", record["heldout"]["accuracy"], "forecast", forecast, flush=True)
    all_probes = [p for r in findings for p in r["probes"]]
    eligible = [p for p in all_probes if p["sign_eligible"]]
    pairs = []
    for small in findings:
        if small["width"] != 16:
            continue
        large = next(r for r in findings if r["prime"] == small["prime"] and r["seed"] == small["seed"] and r["width"] == 64)
        pairs.append({"prime": small["prime"], "seed": small["seed"],
                      "width16": small["early_relative_progress"], "width64": large["early_relative_progress"],
                      "larger_improves": large["early_relative_progress"] > small["early_relative_progress"]})
    maximum_error = max(p["evaluation_identity_max_error"] for p in all_probes)
    correct = sum(p["sign_correct"] for p in eligible)
    summary = {"H1_pass": maximum_error <= 1e-10, "H1_max_error": maximum_error,
               "H2_eligible": len(eligible), "H2_correct": correct,
               "H2_pass": bool(eligible) and correct / len(eligible) >= .9,
               "H3_predictions": sum(r["forecast_success99"] for r in findings),
               "H3_false_positives": sum(r["false_positive"] for r in findings),
               "H3_false_negatives": sum(r["false_negative"] for r in findings),
               "H4_pairs": pairs, "H4_pass": sum(p["larger_improves"] for p in pairs) >= 4,
               "success99_count": sum(r["success99"] for r in findings)}
    report = {"script_sha256": digest(Path(__file__)), "definitions_sha256": digest(ROOT / "experiment.py"),
              "hypotheses_sha256": digest(ROOT / "hypotheses.json"), "python": platform.python_version(),
              "torch": torch.__version__, "arithmetic": "CPU float64; RMS epsilon 2^-23",
              "scope": hypotheses["limits"], "summary": summary, "runs": findings}
    (ROOT / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    with (ROOT / "curves.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["label", "step", "split", "loss", "accuracy"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(curves)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
