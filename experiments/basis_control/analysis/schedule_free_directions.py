"""Task166's immutable-state probe; see schedule_free_directions.md.

Source: arXiv:2405.15682 §2/§4.4 and ResidualScheduleFreeAdamW atccf91e8e.
Compute the real update decomposition in FP64 from an eager CPU FP32 gradient;
do not claim compiled CUDA replay or perform an optimizer step.
"""

import hashlib
import json
from pathlib import Path
import platform

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from lab.domain.optimizers import clipping
from lab.domain.spec import describe
from lab.infrastructure.benchmarks import build_task
from lab.infrastructure.benchmarks.samplers import gather
from lab.infrastructure.loader import load
from lab.infrastructure.nn import build_model
from lab.infrastructure.optim.residual_schedule_free import ResidualScheduleFreeAdamW
from lab.infrastructure.provenance import code

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()


def directions(parameter, gradient, state, group):
    """Return averaging, fresh-gradient and decay terms without mutation."""
    if not group["training"] or group["warmup"]:
        raise ValueError("This diagnostic requires working weights after warmup")
    y, g = parameter.detach().double(), gradient.detach().double()
    z, v = state["base"].double(), state["second"].double()
    beta1, beta2 = group["betas"]
    t, rate = group["step"] + 1, group["lr"]
    maximum = max(rate, group["maximum"])
    total = group["total"] + maximum**2
    c = maximum**2 / total if total else 0.
    v = (beta2 * v + (1 - beta2) * g.square()) / (1 - beta2**t)
    scale = -rate * (1 - beta1 + beta1 * c)
    return c * (z - y), scale * g / (v.sqrt() + group["eps"]), scale * group["weight_decay"] * y


def contributions(gradient, terms):
    g = gradient.detach().double().flatten()
    a, d, w = (value.flatten() for value in terms)
    total, direct = a + d + w, d + w
    norm = lambda value: value.norm().item()
    dot = lambda value: torch.dot(g, value).item()
    fresh_drop = -dot(d)
    denominator = norm(a) + norm(direct)
    return {"gradient_l2": norm(g), "averaging_l2": norm(a), "fresh_l2": norm(d),
            "decay_l2": norm(w), "total_l2": norm(total),
            "gradient_dot_averaging": dot(a), "gradient_dot_fresh": dot(d),
            "gradient_dot_decay": dot(w), "gradient_dot_total": dot(total),
            "averaging_opposition_fraction": dot(a) / fresh_drop if fresh_drop else None,
            "strong_local_opposition": dot(a) >= .5 * fresh_drop if fresh_drop else None,
            "cancellation_fraction": 1 - norm(total) / denominator if denominator else None}


def probe(study_name, label):
    folder = ROOT / "experiments" / study_name
    declaration = json.loads((folder / "protocol.json").read_text())
    if declaration["python"] != platform.python_version() or declaration["torch"] != str(torch.__version__):
        raise ValueError("Frozen runtime changed")
    expected = next(c for c in declaration["cases"] if c["label"] == label)
    phase = json.loads((folder / "runs/phase.json").read_text())
    if phase["status"] != "learning_closed":
        raise ValueError("Only closed states may be probed")
    live = code()
    # Dispatch additions are allowed; all existing model implementations and
    # the gradient, loss, sampler and optimizer operations must be unchanged.
    relevant = {name: digest for name, digest in declaration["engine_sources"].items()
                if (name.startswith(("infrastructure/nn/", "lean_optim/",
                                     "infrastructure/benchmarks/synthetic/"))
                    or name in ("infrastructure/benchmarks/samplers.py",
                                "infrastructure/optim/residual_schedule_free.py"))
                and name != "infrastructure/nn/__init__.py"}
    for name, digest in relevant.items():
        if live[name] != digest:
            raise ValueError(f"Frozen operation changed: {name}")
    spec = load(folder).experiments[label]
    if describe(spec) != expected["experiment"] or clipping(spec.optimizer) != float("inf"):
        raise ValueError("Recipe changed or clipping needs a different decomposition")
    path = folder / "runs" / label / "checkpoint.pt"
    digest = sha(path)
    if digest != phase["cases"][label]["checkpoint_sha256"]:
        raise ValueError("Closed checkpoint changed")
    state = torch.load(path, map_location="cpu", weights_only=True)
    task = build_task(spec.benchmark, spec.seeds.data, torch.device("cpu"))
    fingerprints = expected["data"]["fingerprints"] if study_name == "basis_residual" else expected["fingerprints"]
    if task.fingerprints != fingerprints:
        raise ValueError("Data changed")
    model = build_model(spec.model, task.vocab, spec.seeds.model)
    optimizer = ResidualScheduleFreeAdamW(model.parameters(), **vars(spec.optimizer))
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    group = optimizer.param_groups[0]
    if len(optimizer.param_groups) != 1 or group["step"] != state["step"]:
        raise ValueError("Unexpected optimizer state")
    sampler = task.sampler(spec.budget.batch, spec.seeds.batch_seed)
    sampler.restore(state)
    indices = gather(sampler.next()[0])
    model.train()
    with sdpa_kernel(SDPBackend.MATH):
        logits, targets = task.forward(model, task.inputs(indices, static=True), supervised=True)
        loss = task.loss(logits, targets)
        loss.backward()
    per_parameter, all_gradients, all_terms = {}, [], [[], [], []]
    for name, parameter in model.named_parameters():
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            raise ValueError(f"Missing or nonfinite gradient: {name}")
        terms = directions(parameter, parameter.grad, optimizer.state[parameter], group)
        per_parameter[name] = contributions(parameter.grad, terms)
        all_gradients.append(parameter.grad.detach().flatten().double())
        for collected, term in zip(all_terms, terms, strict=True):
            collected.append(term.flatten())
    aggregate = contributions(torch.cat(all_gradients), tuple(torch.cat(t) for t in all_terms))
    if sha(path) != digest:
        raise ValueError("Probe changed saved checkpoint bytes")
    history = [json.loads(line) for line in (path.parent / "history.jsonl").read_text().splitlines()]
    validation = history[-1]["validation"]
    return {"study": study_name, "label": label, "step": state["step"],
            "checkpoint_sha256": digest, "protocol_sha256": sha(folder / "protocol.json"),
            "checked_operation_sources": relevant, "batch_indices": indices.tolist(),
            "batch_indices_sha256": hashlib.sha256(indices.numpy().tobytes()).hexdigest(),
            "training_batch_loss": loss.item(), "saved_validation_sequence": validation["sequence_accuracy"],
            "saved_validation_token": validation["token_accuracy"],
            "optimizer_group": {k: v for k, v in group.items() if k != "params"},
            "aggregate": aggregate, "parameters": per_parameter}


def main():
    torch.set_num_threads(1)
    cases = []
    for study, prefix in (("basis_control", "gptmini-hard-recall"),
                          ("basis_residual", "canonical-large-recall"),
                          ("attention_temperature", "temperature-hard-recall")):
        for seed in (0, 1):
            result = probe(study, f"{prefix}-s{seed}")
            cases.append(result)
            print(result["label"], json.dumps(result["aggregate"]), flush=True)
    output = {"scope": "One saved next training batch per closed endpoint; CPU only, no optimizer step or final test.",
              "python": platform.python_version(), "torch": str(torch.__version__),
              "probe_source_sha256": sha(Path(__file__)),
              "prediction_sha256": sha(HERE / "schedule_free_directions.md"), "cases": cases}
    destination = HERE / "schedule_free_directions.json"
    text = json.dumps(output, indent=2, allow_nan=False) + "\n"
    if destination.exists() and destination.read_text() != text:
        raise ValueError("Probe no longer reproduces its recorded result")
    destination.write_text(text)


if __name__ == "__main__":
    main()
