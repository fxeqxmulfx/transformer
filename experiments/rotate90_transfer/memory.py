"""Counterfactual Schedule-Free averaging with the actual saved optimizer state.

Source: review/flat_spectrum.md, Optimizer memory, M1--M4. No new fit or
optimizer is introduced. At zero rate the base remains fixed, while its
positive historical maximum continues to weight averages. We evaluate that
exact closed-form path with CUDA max-autotune, and independently check it
against 32 real-gradient source updates in CPU float64. The latter retains
the original float32 normalization epsilon. Saved checkpoints stay immutable.
"""

import hashlib
import json
from pathlib import Path
import platform

import torch
from torch.nn import functional as F

from lab.infrastructure.benchmarks import build_task
from lab.infrastructure.benchmarks.samplers import gather
from lab.infrastructure.loader import load
from lab.infrastructure.nn import build_model
from lab.infrastructure.optim import build_optimizer

ROOT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def restore(spec, checkpoint):
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = build_model(spec.model, spec.benchmark.prime, spec.seeds.model).double()
    model.normalize = lambda x: F.rms_norm(x, (x.size(-1),), eps=torch.finfo(torch.float32).eps)
    model.load_state_dict(state["model"])
    optimizer = build_optimizer(spec.optimizer, model)
    optimizer.load_state_dict(state["optimizer"])
    assert optimizer.param_groups[0]["training"]
    return model, optimizer, state


@torch.no_grad()
def coordinates(optimizer):
    group = optimizer.param_groups[0]
    beta = group["betas"][0]
    assert group["training"]
    y = [p.detach().clone() for p in group["params"]]
    z = [optimizer.state[p]["base"].clone() for p in group["params"]]
    x = [a + (1 - 1 / beta) * (b - a) for a, b in zip(y, z, strict=True)]
    return x, y, z


@torch.no_grad()
def zero_rate_point(x, z, total, maximum, updates):
    """M2: the exact evaluation point after n zero-rate source updates."""
    if updates < 0 or total <= 0 or maximum <= 0:
        raise ValueError("The declared continuation requires a positive stored total and maximum")
    alpha = total / (total + updates * maximum ** 2)
    return [b + alpha * (a - b) for a, b in zip(x, z, strict=True)]


def check_source(spec, checkpoint, updates=32):
    model, optimizer, state = restore(spec, checkpoint)
    task = build_task(spec.benchmark, spec.seeds.data, torch.device("cpu"))
    sampler = task.sampler(spec.budget.batch, spec.seeds.batch_seed)
    sampler.restore(state)
    group = optimizer.param_groups[0]
    x, y, z = coordinates(optimizer)
    total, maximum = group["total"], group["maximum"]
    predicted = zero_rate_point(x, z, total, maximum, updates)
    second = [optimizer.state[p]["second"].clone() for p in model.parameters()]
    group["lr"] = 0.
    for _ in range(updates):
        batch = task.inputs(gather(sampler.next()[0]))
        model.zero_grad(set_to_none=True)
        F.cross_entropy(model(batch[:, :2]), batch[:, 2]).backward()
        optimizer.step()
    optimizer.eval()
    with torch.no_grad():
        error = max(float((p - expected).abs().max())
                    for p, expected in zip(model.parameters(), predicted, strict=True))
        base_error = max(float((optimizer.state[p]["base"] - before).abs().max())
                         for p, before in zip(model.parameters(), z, strict=True))
        second_change = sum(float((optimizer.state[p]["second"] - before).square().sum())
                            for p, before in zip(model.parameters(), second, strict=True)) ** .5
        motion = sum(float((a - b).square().sum()) for a, b in zip(x, predicted, strict=True)) ** .5
    return {"updates": updates, "max_coordinate_error": error, "base_max_change": base_error,
            "second_moment_change_norm": second_change, "evaluation_movement_norm": motion,
            "total_error": abs(group["total"] - (total + updates * maximum ** 2)),
            "next_sampler_cursor": sampler.cursor}


class CompiledEvaluation:
    """Reuse one actual architecture's compiled inference across checkpoint states."""
    def __init__(self, spec):
        torch._dynamo.reset()
        torch._dynamo.config.recompile_limit = 64
        self.model = build_model(spec.model, spec.benchmark.prime, spec.seeds.model).cuda()
        self.forward = torch.compile(self.model, mode="max-autotune", fullgraph=True, dynamic=False)
        self.task = build_task(spec.benchmark, spec.seeds.data, torch.device("cuda"))

    @torch.no_grad()
    def install(self, values):
        for p, value in zip(self.model.parameters(), values, strict=True):
            p.copy_(value)

    @torch.no_grad()
    def metrics(self):
        result = {}
        for name, rows in self.task.splits.items():
            loss, correct, strict = 0., 0, 0
            for chunk in rows.split(1024):
                torch.compiler.cudagraph_mark_step_begin()
                logits = self.forward(chunk[:, :2])
                loss += float(F.cross_entropy(logits, chunk[:, 2], reduction="sum"))
                correct += int((logits.argmax(-1) == chunk[:, 2]).sum())
                target = logits.gather(1, chunk[:, 2:3])
                strict += int(((logits >= target).sum(-1) == 1).sum())
            result[name] = {"loss": loss / len(rows), "accuracy": correct / len(rows),
                            "strict_winner_fraction": strict / len(rows), "examples": len(rows)}
        return result

    @torch.no_grad()
    def logits(self, rows):
        outputs = []
        for chunk in rows.split(1024):
            torch.compiler.cudagraph_mark_step_begin()
            outputs.append(self.forward(chunk[:, :2]).cpu().clone())
        return torch.cat(outputs)


def main():
    torch.set_num_threads(1)
    hypotheses = json.loads((ROOT / "memory_hypotheses.json").read_text())
    specs = load(ROOT).experiments
    records = []
    # Reusing compiled shapes changes neither checkpoint selection nor thresholds.
    sources = sorted(hypotheses["sources"], key=lambda v: (specs[v["label"]].benchmark.prime,
                       specs[v["label"]].model.width, v["label"], v["step"]))
    evaluator, previous = None, None
    for source in sources:
        label, step = source["label"], source["step"]
        spec = specs[label]
        path = ROOT / "runs" / label / "checkpoints" / f"step-{step:08d}.pt"
        assert digest(path) == source["sha256"]
        model, optimizer, state = restore(spec, path)
        key = spec.benchmark.prime, spec.model.width
        if key != previous:
            evaluator = CompiledEvaluation(spec)
            previous = key
        x, y, z = coordinates(optimizer)
        group = optimizer.param_groups[0]
        roles = {}
        for name, values in (("evaluation", x), ("query", y), ("base", z)):
            evaluator.install(values)
            roles[name] = evaluator.metrics()
        curve = []
        for n in hypotheses["zero_rate_updates"]:
            evaluator.install(zero_rate_point(x, z, group["total"], group["maximum"], n))
            curve.append({"updates": n, "metrics": evaluator.metrics()})
        evaluator.install(x)
        positive = evaluator.logits(evaluator.task.splits["heldout"])
        evaluator.install([-value for value in x])
        negative = evaluator.logits(evaluator.task.splits["heldout"])
        symmetry_error = float((positive - negative).abs().max())
        evaluator.install([torch.zeros_like(value) for value in x])
        midpoint = evaluator.metrics()
        direct = check_source(spec, path, hypotheses["source_check_updates"])
        assert digest(path) == source["sha256"]
        record = {**source, "prime": spec.benchmark.prime, "exponent": spec.benchmark.exponent,
                  "width": spec.model.width, "seed": spec.seeds.model, "roles": roles,
                  "stored_total": group["total"], "stored_maximum": group["maximum"],
                  "zero_rate_curve": curve, "source_check": direct,
                  "negation_max_logit_error": symmetry_error, "antipodal_midpoint": midpoint,
                  "task": evaluator.task.metadata}
        records.append(record)
        print(label, step, "start/base/final accuracy",
              roles["evaluation"]["heldout"]["accuracy"], roles["base"]["heldout"]["accuracy"],
              curve[-1]["metrics"]["heldout"]["accuracy"], "source error", direct["max_coordinate_error"], flush=True)
    selected = [r for r in records if r["width"] == 64 and r["prime"] in (97, 193) and r["step"] == 750]
    summary = {
        "H1_max_coordinate_error": max(r["source_check"]["max_coordinate_error"] for r in records),
        "H1_max_base_change": max(r["source_check"]["base_max_change"] for r in records),
        "H2_moved": sum(r["source_check"]["evaluation_movement_norm"] > 1e-12 for r in records),
        "H3_cases": [{"label": r["label"],
                      "initial_improvement": r["zero_rate_curve"][1]["metrics"]["heldout"]["loss"] < r["zero_rate_curve"][0]["metrics"]["heldout"]["loss"],
                      "final_loss_exceeds_initial": r["zero_rate_curve"][-1]["metrics"]["heldout"]["loss"] > r["zero_rate_curve"][0]["metrics"]["heldout"]["loss"],
                      "final_accuracy_below_half": r["zero_rate_curve"][-1]["metrics"]["heldout"]["accuracy"] < .5}
                     for r in selected],
        "H4_max_negation_logit_error": max(r["negation_max_logit_error"] for r in records),
        "H4_max_midpoint_strict_fraction": max(r["antipodal_midpoint"]["heldout"]["strict_winner_fraction"] for r in records),
    }
    report = {"script_sha256": digest(Path(__file__)), "hypotheses_sha256": digest(ROOT / "memory_hypotheses.json"),
              "source_setup_commit": hypotheses["source_setup_commit"], "python": platform.python_version(),
              "torch": torch.__version__, "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(),
              "scope": hypotheses["scope"], "summary": summary, "checkpoints": records}
    (ROOT / "memory_results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
