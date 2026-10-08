"""Archived AdamW buffers versus actual exhaustive objective gradients.

Source: lab.infrastructure.engine.eager.EagerStepper, optim.adamw,
benchmarks.samplers.EpochSampler and domain.training.rate at 91bb895;
PyTorch 2.14.1 AdamW moment/bias-correction equations. Lean comparison:
Transformer.Grokking.AdamW.SecondStep.second_step_loss_deriv and
MomentumCounterexample.quadratic_second_loss_increases.
Deviations: a restored next minibatch drives one disposable CPU clone;
float64 diagnostics read float32 autograd and native optimizer outputs.
Exhaustive train/held-out gradients and separate answer/EOS losses are
diagnostics, never gradients supplied to that update. A finite probe step
is not a continuation of the archived CUDA trajectory or its budget.
"""

import copy
import hashlib
import math

import torch

from lab.domain.optimizers import clipping
from lab.domain.training import rate
from lab.infrastructure.benchmarks.samplers import gather
from lab.infrastructure.optim import build_optimizer

from probes.capture import evaluating
from probes.gradients import cosine


def flatten(values):
    return torch.cat([value.detach().cpu().double().reshape(-1) for value in values])


def scopes(task):
    zero = task.corpus.tokens.index("0")
    return {f"{split}_{kind}": rows if kind == "all" else rows[rows[:, 5] != zero]
            for split, rows in task.splits.items() for kind in ("all", "nonzero")}


def gradients(model, task, rows, batch=512):
    """Exhaustive sample-weighted means, preserving existing .grad buffers."""
    if len(rows) == 0 or batch < 1:
        raise ValueError("Exhaustive gradients require nonempty rows and positive chunk size")
    parameters = [p for p in model.parameters() if p.requires_grad]
    collected = {key: torch.zeros(sum(p.numel() for p in parameters), dtype=torch.float64)
                 for key in ("full", "answer", "EOS")}
    sums = {key: 0. for key in collected}
    correct = 0
    with evaluating(model), torch.enable_grad():
        for chunk in rows.split(batch):
            output, targets = task.forward(model, chunk)
            if targets.shape != (len(chunk), 2) or output.shape[:2] != targets.shape:
                raise ValueError("Expected answer and EOS at the two original supervised positions")
            position = task.position_losses(output, targets)
            losses = {"full": task.loss(output, targets), "answer": position[0], "EOS": position[1]}
            weight = len(chunk) / len(rows)
            for key in ("full", "answer", "EOS"):
                observed = torch.autograd.grad(losses[key], parameters, allow_unused=True, retain_graph=key != "EOS")
                collected[key].add_(flatten([torch.zeros_like(p) if g is None else g
                                            for p, g in zip(parameters, observed)]), alpha=weight)
                sums[key] += float(losses[key].detach()) * weight
            correct += int((output[:, 0].argmax(-1) == targets[:, 0]).sum())
    if not all(torch.isfinite(value).all() for value in collected.values()):
        raise FloatingPointError("Nonfinite exhaustive gradient")
    return {"examples": len(rows), "losses": sums, "answer_accuracy": correct / len(rows)}, collected


@torch.no_grad()
def losses(model, task, rows, batch=512):
    """Same row weighting and supervision as the exhaustive gradient reader."""
    sums = {key: 0. for key in ("full", "answer", "EOS")}
    correct = 0
    with evaluating(model):
        for chunk in rows.split(batch):
            output, targets = task.forward(model, chunk)
            position = task.position_losses(output, targets)
            for key, value in (("full", task.loss(output, targets)), ("answer", position[0]), ("EOS", position[1])):
                sums[key] += float(value) * len(chunk) / len(rows)
            correct += int((output[:, 0].argmax(-1) == targets[:, 0]).sum())
    return {"examples": len(rows), "losses": sums, "answer_accuracy": correct / len(rows)}


@torch.no_grad()
def retained_direction(optimizer, expected_step):
    """Direction encoded by current buffers; no new gradient is inserted."""
    result, decays = [], []
    for group in optimizer.param_groups:
        if group["amsgrad"] or group["maximize"]:
            raise ValueError("Only the archived ordinary minimizing AdamW is supported")
        b1, b2 = group["betas"]
        if not (0 <= b1 < 1 and 0 <= b2 < 1 and group["eps"] > 0):
            raise ValueError("Native bias correction requires valid betas and positive epsilon")
        for parameter in group["params"]:
            state = optimizer.state[parameter]
            if set(state) != {"step", "exp_avg", "exp_avg_sq"} or float(state["step"]) != expected_step:
                raise ValueError("Every unique parameter must retain its actual completed-update clock")
            m, v = state["exp_avg"].double(), state["exp_avg_sq"].double()
            if not torch.isfinite(m).all() or not torch.isfinite(v).all() or (v < 0).any():
                raise FloatingPointError("Invalid archived native moment buffers")
            result.append((m / (1 - b1 ** expected_step)) / ((v / (1 - b2 ** expected_step)).sqrt() + group["eps"]))
            decays.extend([group["weight_decay"]] * parameter.numel())
    return flatten(result), torch.tensor(decays, dtype=torch.float64)


def alignment(gradient, directions):
    return {name: {"gradient_dot_down_direction": float(gradient @ direction),
                   "loss_rate_derivative": -float(gradient @ direction),
                   "cosine": cosine(gradient, direction)} for name, direction in directions.items()}


def measure(model, task, spec, snapshot, batch=512):
    """Preserve the supplied model/snapshot and global RNG, including failures."""
    if snapshot["step"] < 1 or any(p.device.type != "cpu" for p in model.parameters()):
        raise ValueError("Archived noninitial native states and a CPU model are required")
    with evaluating(model), torch.random.fork_rng(devices=[]):
        clone = copy.deepcopy(model)
        parameters = [(name, p) for name, p in clone.named_parameters() if p.requires_grad]
        optimizer = build_optimizer(spec.optimizer, clone)
        if [id(p) for g in optimizer.param_groups for p in g["params"]] != [id(p) for _, p in parameters]:
            raise ValueError("Direction analysis requires native groups in model registration order")
        expected_ids = [g["params"] for g in optimizer.state_dict()["param_groups"]]
        if [g["params"] for g in snapshot["optimizer"]["param_groups"]] != expected_ids:
            raise ValueError("Archived parameter registration order differs from the native builder")
        optimizer.load_state_dict(copy.deepcopy(snapshot["optimizer"]))
        old, decays = retained_direction(optimizer, snapshot["step"])
        before = flatten([p for _, p in parameters])
        populations = scopes(task)
        observed = {name: gradients(clone, task, rows, batch) for name, rows in populations.items()}
        sampler = task.sampler(spec.budget.batch, spec.seeds.batch_seed)
        sampler.restore(copy.deepcopy(snapshot))
        parts, place = sampler.next()
        indices = gather(parts)
        learning_rate = rate(spec.optimizer.lr, spec.schedule, snapshot["step"])
        if not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("The finite diagnostic displacement requires a positive scheduled rate")
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        clone.train()
        optimizer.zero_grad(set_to_none=True)
        output, target = task.forward(clone, task.inputs(indices))
        task.loss(output, target).backward()
        raw_norm = torch.nn.utils.clip_grad_norm_(clone.parameters(), clipping(spec.optimizer))
        if not torch.isfinite(raw_norm):
            raise FloatingPointError("Nonfinite restored-minibatch gradient")
        stochastic = flatten([p.grad if p.grad is not None else torch.zeros_like(p) for _, p in parameters])
        optimizer.step()
        adaptive, after_decays = retained_direction(optimizer, snapshot["step"] + 1)
        if not torch.equal(decays, after_decays):
            raise RuntimeError("Unexpected change to the archived decay groups")
        direction = adaptive + decays * before
        empirical = (before - flatten([p for _, p in parameters])) / learning_rate
        directions = {"retained_buffer": old, "new_adaptive": adaptive, "decay": decays * before,
                      "total_algorithm": direction, "finite_CPU_displacement": empirical}
        result = {"protocol": "restored_next_minibatch; disposable_native_CPU_step; exhaustive_current_gradients",
                  "precision": f"float64_statistics_of_{parameters[0][1].dtype}_CPU_autograd_and_native_step; not_certified",
                  "scope": "counterfactual_observation_only; archived_CUDA_training_not_resumed",
                  "step_before": snapshot["step"], "rate": learning_rate,
                  "parameters": [{"name": name, "shape": list(p.shape)} for name, p in parameters],
                  "minibatch": {"examples": len(indices), "place": place,
                                "indices_sha256": hashlib.sha256(indices.numpy().astype("<i8").tobytes()).hexdigest(),
                                "raw_gradient_l2": float(raw_norm), "clipped_gradient_l2": float(stochastic.norm())},
                  "direction_l2": {key: float(value.norm()) for key, value in directions.items()},
                  "finite_displacement_algorithm_relative_error": float((empirical - direction).norm() / direction.norm())
                     if float(direction.norm()) > 0 else None,
                  "next_minibatch_alignment": alignment(stochastic, directions), "populations": {}}
        for name, (baseline, component_gradients) in observed.items():
            result["populations"][name] = {"before": baseline, "after": losses(clone, task, populations[name], batch),
                "gradient_l2": {key: float(value.norm()) for key, value in component_gradients.items()},
                "alignment": {key: alignment(value, directions) for key, value in component_gradients.items()},
                "full_gradient_component_relative_error":
                    float((component_gradients["full"] - (component_gradients["answer"] + component_gradients["EOS"]) / 2).norm()
                          / ((component_gradients["answer"].norm() + component_gradients["EOS"].norm()) / 2))
                    if float(component_gradients["answer"].norm() + component_gradients["EOS"].norm()) > 0 else None}
    return result
