"""Frozen gradient/decay controls and the actual-source response decomposition."""

import hashlib
import json
from pathlib import Path
import platform

import torch

from lab.infrastructure.loader import load
from lab.infrastructure.store import RunDirectory

from dynamics import (MEMORY, apply_matrix, check_source, coordinates, evaluation,
                      homogeneous, plane_residual, restore)

ROOT = Path(__file__).resolve().parent
TRANSFER = ROOT.parent / "rotate90_transfer"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sampler_equal(a, b):
    return (a["cursor"] == b["cursor"] and a["rotary_random_state"] == b["rotary_random_state"]
            and torch.equal(a["permutation"], b["permutation"]))


def initial_check(pinned, initial):
    assert pinned["step"] == initial["step"] == 400
    assert sampler_equal(pinned, initial)
    for key, value in pinned["model"].items():
        assert torch.equal(value, initial["model"][key])
    for index, state in pinned["optimizer"]["state"].items():
        for key, value in state.items():
            assert torch.equal(value, initial["optimizer"]["state"][index][key])
    for key in ("step", "total", "maximum", "betas", "eps", "warmup", "training"):
        assert pinned["optimizer"]["param_groups"][0][key] == initial["optimizer"]["param_groups"][0][key]


def main():
    torch.set_num_threads(1)
    hypotheses = json.loads((ROOT / "hypotheses.json").read_text())
    specs = load(ROOT).experiments
    records = []
    evaluator, previous = None, None
    sources = sorted(hypotheses["sources"], key=lambda s: (
        specs[f"full-{s['label']}"].benchmark.prime, specs[f"full-{s['label']}"].model.width, s["label"]))
    for source in sources:
        label = source["label"]
        spec = specs[f"full-{label}"]
        path = TRANSFER / "runs" / label / "checkpoints/step-00000400.pt"
        assert digest(path) == source["sha256"]
        model, optimizer, pinned = restore(spec, path)
        group = optimizer.param_groups[0]
        x, y, z = coordinates(optimizer)
        key = spec.benchmark.prime, spec.model.width
        if key != previous:
            evaluator = MEMORY["CompiledEvaluation"](spec)
            previous = key
        curve = []
        for n in hypotheses["zero_gradient_horizons"]:
            matrix = homogeneous(group, n)
            point = evaluation(*apply_matrix(matrix, y, z), group["betas"][0])
            evaluator.install(point)
            curve.append({"updates": n, "matrix": matrix, "metrics": evaluator.metrics(),
                          "plane": plane_residual(point, y, z)})
        evaluator.install(MEMORY["zero_rate_point"](x, z, group["total"], group["maximum"], 1100))
        zero_rate = evaluator.metrics()
        arms, finals = {}, []
        for arm in ("full", "no-decay"):
            run = RunDirectory(ROOT / "runs" / f"{arm}-{label}")
            final = run.checkpoint("cpu")
            assert final["step"] == 1500
            initial = torch.load(run.path / "checkpoints/step-00000400.pt", map_location="cpu", weights_only=True)
            initial_check(pinned, initial)
            _, opt, _ = restore(spec, run.path / "checkpoint.pt")
            point, _, _ = coordinates(opt)
            history = run.records("history")
            arms[arm] = {"history": history, "endpoint": history[-1],
                         "plane": plane_residual(point, y, z),
                         "checkpoint_sha256": digest(run.path / "checkpoint.pt")}
            finals.append(final)
        assert sampler_equal(*finals)
        original = RunDirectory(TRANSFER / "runs" / label).records("history")[-1]
        actual = arms["full"]["endpoint"]["heldout"]
        difference = {k: abs(actual[k] - original["heldout"][k]) for k in ("accuracy", "loss")}
        direct = check_source(spec, path, hypotheses["source_check_updates"])
        assert digest(path) == source["sha256"]
        record = {**source, "prime": spec.benchmark.prime, "width": spec.model.width,
                  "exponent": spec.benchmark.exponent, "arms": arms,
                  "source_replay_difference": difference, "source_check": direct,
                  "zero_gradient_curve": curve, "zero_rate_endpoint": zero_rate,
                  "initial_state_exact": True, "matched_final_sampler": True}
        records.append(record)
        print(label, "full/no-decay/zero-gradient accuracy", actual["accuracy"],
              arms["no-decay"]["endpoint"]["heldout"]["accuracy"],
              curve[-1]["metrics"]["heldout"]["accuracy"], flush=True)
    selected = [r for r in records if r["width"] == 64 and r["prime"] in (97, 193)]
    summary = {
        "H1_cases": [{"label": r["label"], **r["source_replay_difference"],
                      "pass": r["source_replay_difference"]["accuracy"] <= .002 and
                              r["source_replay_difference"]["loss"] <= .01} for r in records],
        "H2_max_source_error": max(max(r["source_check"][k] for k in
            ("forced_buffer_max_error", "homogeneous_buffer_max_error", "evaluation_max_error")) for r in records),
        "H2_max_plane_residual": max(r["source_check"]["zero_gradient_plane"]["relative_residual"] for r in records),
        "H3_cases": [{"label": r["label"], "full_above99": r["arms"]["full"]["endpoint"]["heldout"]["accuracy"] > .99,
                      "zero_gradient_below99": r["zero_gradient_curve"][-1]["metrics"]["heldout"]["accuracy"] < .99} for r in selected],
        "H4_cases": [{"label": r["label"], "no_decay_above99": r["arms"]["no-decay"]["endpoint"]["heldout"]["accuracy"] > .99} for r in selected],
        "H5_cases": [{"label": r["label"], **r["arms"]["full"]["plane"],
                      "pass": r["arms"]["full"]["plane"]["relative_residual"] > 1e-4} for r in records],
    }
    result = {"script_sha256": digest(Path(__file__)), "dynamics_sha256": digest(ROOT / "dynamics.py"),
              "hypotheses_sha256": digest(ROOT / "hypotheses.json"), "python": platform.python_version(),
              "torch": torch.__version__, "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(),
              "scope": hypotheses["scope"], "summary": summary, "sources": records}
    (ROOT / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
