"""Describe every frozen task145 endpoint without training or opening final test.

This post hoc mechanism audit uses the first 256 training rows at the saved
Schedule-Free training iterate. It measures true CPU FP32 CE derivatives and
last-token depth weights. Gradient size alone does not prove optimizer stalling
or predict later generalization. No result selects a checkpoint or recipe.
"""

import hashlib
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from lab.infrastructure.benchmarks import build_task
from lab.infrastructure.loader import load
from lab.infrastructure.nn import build_model
from lab.infrastructure.provenance import code
from lab.infrastructure.store import RunDirectory

HERE = Path(__file__).resolve().parents[1] / "experiments/residual_generalization"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def depth_weights(mixer, values):
    """Read the declared scores independently; never replace the model forward."""
    keys = F.rms_norm(values, (mixer.width,), eps=mixer.eps)
    if mixer.mixing == "attnres":
        scores = (keys * mixer.key_gain * mixer.query).sum(-1)
    else:
        physical = mixer.query * getattr(mixer, "query_scale", 1.)
        query = F.rms_norm(physical, (mixer.width,), eps=mixer.eps)
        scores = (keys * query).sum(-1) / math.sqrt(mixer.width)
    return scores.softmax(-1)


def normalization_statistics(module, inputs, output, distinct):
    """Local RMS VJP, excluding gradients that reach an input through other paths.

    For r=sqrt(mean(x²)+eps), Jᵀg=g/r-x*mean(x*g)/r³. Evaluate this
    derivative in FP64 at the retained FP32 values and upstream adjoint.
    The epsilon is the actual FP32 forward epsilon, including at zero.
    GPTMini.rmsNormEps_lipschitz bounds sensitivity, not a gradient lower bound.
    """
    if output.grad is None:
        raise AssertionError("A normalization output has no retained adjoint")
    x, g = inputs[..., -1, :].double(), output.grad.detach()[..., -1, :].double()
    eps = module.eps if module.eps is not None else torch.finfo(inputs.dtype).eps
    upstream = g.norm(dim=-1)
    if module.weight is not None:
        g = g * module.weight.detach().double()
    square = x.square().mean(-1, keepdim=True)
    radius = (square + eps).sqrt()
    local = g / radius - x * (x * g).mean(-1, keepdim=True) / radius.pow(3)
    # Independently check this local formula against autograd, without another
    # model forward or backward through shared residual inputs.
    reference_input = x.detach().requires_grad_(True)
    reference = F.rms_norm(reference_input, (x.shape[-1],), eps=eps)
    expected = torch.autograd.grad(reference, reference_input, g)[0]
    torch.testing.assert_close(local, expected, rtol=1e-8, atol=1e-12)
    ratio = local.norm(dim=-1) / upstream.clamp_min(torch.finfo(torch.float64).tiny)

    def summarize(mask):
        present = upstream[mask] > 0
        values = ratio[mask][present]
        return {"rows": int(mask.sum()), "zero_upstream_fraction": float((~present).double().mean()),
                "epsilon_dominated_fraction": float((square[mask] <= eps).double().mean()),
                "median_input_rms": float(square[mask].sqrt().median()),
                "maximum_tangent_gain": float(radius[mask].reciprocal().max()),
                "median_actual_vjp_gain": float(values.median()) if len(values) else None,
                "maximum_actual_vjp_gain": float(values.max()) if len(values) else None}

    return {"eps": eps, "all": summarize(torch.ones(len(x), dtype=torch.bool)),
            "distinct_operands": summarize(distinct)}


def describe_case(declaration, spec):
    run = RunDirectory(HERE / "runs" / declaration["label"])
    checkpoint = run.checkpoint("cpu")
    if checkpoint is None:
        return {"label": declaration["label"], "status": "no_checkpoint"}
    checkpoint_hash = sha(run.path / "checkpoint.pt")
    if not checkpoint["optimizer"]["param_groups"][0]["training"]:
        raise ValueError("Expected the saved Schedule-Free training iterate")
    task = build_task(spec.benchmark, spec.seeds.data, torch.device("cpu"))
    if task.metadata != declaration["task"]:
        raise ValueError("Frozen data fingerprints changed")
    model = build_model(spec.model, task.vocab, spec.seeds.model)
    model.load_state_dict(checkpoint["model"])
    model.train()
    weights, normalizations = [], []

    def inspect(mixer, args):
        with torch.no_grad():
            value = depth_weights(mixer, args[0][:, -1])
            torch.testing.assert_close(value.sum(-1), torch.ones(len(value)))
            entropy = -(value * value.clamp_min(torch.finfo(value.dtype).tiny).log()).sum(-1)
            weights.append({
                "sources": value.shape[-1],
                "mean_source_weights": value.mean(0).tolist(),
                "mean_normalized_entropy": float(entropy.mean() / math.log(value.shape[-1])),
                "mean_largest_weight": float(value.amax(-1).mean()),
                "fraction_largest_weight_above_0_9": float((value.amax(-1) > .9).float().mean()),
            })

    handles = [m.register_forward_pre_hook(inspect) for m in model.residual_mixers]
    def retain(name, module, args, output):
        output.retain_grad()
        normalizations.append((name, module, args[0].detach(), output))
    handles += [module.register_forward_hook(
        lambda m, args, out, name=name: retain(name, m, args, out))
        for name, module in model.named_modules() if isinstance(module, torch.nn.RMSNorm)]
    rows = task.splits["train"][:256]
    try:
        logits = model(rows[:, :-1])[:, -1]
        loss = F.cross_entropy(logits, rows[:, -1])
        loss.backward()
    finally:
        for handle in handles:
            handle.remove()
    gradients = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()):
            raise FloatingPointError(f"Missing/nonfinite gradient: {name}")
        norm = float(parameter.detach().norm())
        derivative = float(parameter.grad.norm())
        gradients[name] = {"parameter_l2": norm, "gradient_l2": derivative,
                           "gradient_over_parameter_l2": derivative / norm if norm else None}
    if sha(run.path / "checkpoint.pt") != checkpoint_hash:
        raise AssertionError("Audit modified a frozen checkpoint")
    normalization = {name: normalization_statistics(module, value, output, rows[:, 0] != rows[:, 1])
                     for name, module, value, output in normalizations}
    return {"label": declaration["label"], "step": checkpoint["step"],
            "mode": spec.model.mixing, "checkpoint_sha256": checkpoint_hash,
            "training_rows": len(rows), "training_loss": float(loss.detach()),
            "training_accuracy": float((logits.argmax(-1) == rows[:, -1]).float().mean()),
            "gradients": gradients, "last_token_depth_weights": weights,
            "normalization": normalization}


def main():
    phase = json.loads((HERE / "runs/phase.json").read_text())
    if phase["status"] not in ("learning_closed", "closed"):
        raise ValueError("Run endpoint diagnostics only after learning closes")
    protocol = json.loads((HERE / "protocol.json").read_text())
    if protocol["engine_sources"] != code():
        raise ValueError("Frozen engine source changed")
    torch.set_num_threads(1)
    study = load(HERE)
    cases = [describe_case(row, study.experiments[row["label"]]) for row in protocol["cases"]]
    report = {"protocol_sha256": sha(HERE / "protocol.json"),
              "scope": __doc__, "cases": cases}
    (HERE / "mechanism.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Audited {len(cases)} declared endpoints; no optimizer steps or final-test access.")


if __name__ == "__main__":
    main()
