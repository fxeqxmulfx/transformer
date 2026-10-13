"""Same-gradient counterfactuals of actual archived native AdamW moments.

Source: native optimizer and EagerStepper at 0033b1b; PyTorch 2.14.1
AdamW equations. Compare Grokking.AdamW.SecondStep and the proposed
history-forgetting extension. Deviations: a single restored minibatch
supplies the same clipped gradient to four disposable CPU copies. Full
reset also resets the bias clock; separate first/second-moment resets
retain it. Exhaustive objectives evaluate the copies, never train them.
No archived CUDA trajectory, training recipe or budget is modified.
"""

import copy
import hashlib
import math

import torch

from lab.domain.optimizers import clipping
from lab.domain.training import rate
from lab.infrastructure.benchmarks.samplers import gather
from lab.infrastructure.optim import build_optimizer

from momentum_directions import alignment, flatten, gradients, losses, retained_direction, scopes
from probes.capture import evaluating
from probes.gradients import cosine

BRANCHES = ("retained", "fresh_adamw", "reset_first_moment", "reset_second_moment")


@torch.no_grad()
def reset_state(optimizer, branch):
    """Mutate only a disposable optimizer; partial resets retain its clock."""
    if branch not in BRANCHES:
        raise ValueError("Unknown moment-reset branch")
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            state = optimizer.state[parameter]
            if branch in ("fresh_adamw", "reset_first_moment"):
                state["exp_avg"].zero_()
            if branch in ("fresh_adamw", "reset_second_moment"):
                state["exp_avg_sq"].zero_()
            if branch == "fresh_adamw":
                state["step"].zero_()


def restored_optimizer(spec, model, snapshot):
    """Reject inconsistent registration, moments and completed-update clocks."""
    optimizer = build_optimizer(spec.optimizer, model)
    parameters = [p for p in model.parameters() if p.requires_grad]
    if [id(p) for group in optimizer.param_groups for p in group["params"]] != [id(p) for p in parameters]:
        raise ValueError("Native groups must follow unique model registration order")
    ids = [group["params"] for group in optimizer.state_dict()["param_groups"]]
    if [group["params"] for group in snapshot["optimizer"]["param_groups"]] != ids:
        raise ValueError("Archived parameter registration differs")
    optimizer.load_state_dict(copy.deepcopy(snapshot["optimizer"]))
    retained_direction(optimizer, snapshot["step"])
    return optimizer


def measure(model, task, spec, snapshot, batch=512):
    """Preserve caller weights, gradients, modes, optimizer snapshot and RNG."""
    if snapshot["step"] < 1 or any(p.device.type != "cpu" for p in model.parameters()):
        raise ValueError("Actual noninitial archived states and a CPU model are required")
    with evaluating(model), torch.random.fork_rng(devices=[]):
        base = copy.deepcopy(model)
        restored_optimizer(spec, base, snapshot)
        populations = {"train_all": task.splits["train"], "heldout_nonzero": scopes(task)["heldout_nonzero"]}
        observed = {name: gradients(base, task, rows, batch) for name, rows in populations.items()}
        sampler = task.sampler(spec.budget.batch, spec.seeds.batch_seed)
        sampler.restore(copy.deepcopy(snapshot))
        parts, place = sampler.next()
        indices = gather(parts)
        learning_rate = rate(spec.optimizer.lr, spec.schedule, snapshot["step"])
        if not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("Positive scheduled rate is required")
        base.train()
        base.zero_grad(set_to_none=True)
        output, target = task.forward(base, task.inputs(indices))
        task.loss(output, target).backward()
        raw_norm = torch.nn.utils.clip_grad_norm_(base.parameters(), clipping(spec.optimizer))
        if not torch.isfinite(raw_norm):
            raise FloatingPointError("Nonfinite restored-minibatch gradient")
        parameters = [p for p in base.parameters() if p.requires_grad]
        shared = [None if p.grad is None else p.grad.detach().clone() for p in parameters]
        stochastic = flatten([torch.zeros_like(p) if g is None else g for p, g in zip(parameters, shared)])
        before = flatten(parameters)
        result = {"protocol": "one_shared_clipped_gradient; four_disposable_native_CPU_steps",
                  "precision": "float64_statistics_of_native_dtype_gradients_and_CPU_updates; uncertified",
                  "scope": "counterfactual_observation_only; no_training_resumed",
                  "step_before": snapshot["step"], "rate": learning_rate,
                  "minibatch": {"examples": len(indices), "place": place,
                                "indices_sha256": hashlib.sha256(indices.numpy().astype("<i8").tobytes()).hexdigest(),
                                "raw_gradient_l2": float(raw_norm), "clipped_gradient_l2": float(stochastic.norm())},
                  "before": {name: record for name, (record, _) in observed.items()}, "branches": {}}
        native_direction = None
        native_displacement = None
        for branch in BRANCHES:
            clone = copy.deepcopy(model)
            optimizer = restored_optimizer(spec, clone, snapshot)
            reset_state(optimizer, branch)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            copied_parameters = [p for p in clone.parameters() if p.requires_grad]
            for parameter, gradient in zip(copied_parameters, shared):
                parameter.grad = None if gradient is None else gradient.clone()
            optimizer.step()
            clock = 1 if branch == "fresh_adamw" else snapshot["step"] + 1
            adaptive, decays = retained_direction(optimizer, clock)
            direction = adaptive + decays * before
            displacement = (before - flatten(copied_parameters)) / learning_rate
            if not torch.isfinite(displacement).all():
                raise FloatingPointError("Nonfinite counterfactual displacement")
            directions = {"new_adaptive": adaptive, "total_algorithm": direction,
                          "finite_CPU_displacement": displacement}
            if branch == "retained":
                native_direction, native_displacement = direction, displacement
            populations_after = {}
            for name, (_, component_gradients) in observed.items():
                populations_after[name] = {"after": losses(clone, task, populations[name], batch),
                    "alignment": {key: alignment(value, directions) for key, value in component_gradients.items()}}
            native_norm = float(native_direction.norm())
            finite_norm = float(native_displacement.norm())
            result["branches"][branch] = {"clock_after": clock,
                "direction_l2": {key: float(value.norm()) for key, value in directions.items()},
                "finite_displacement_algorithm_relative_error": float((displacement - direction).norm() / direction.norm())
                    if float(direction.norm()) > 0 else None,
                "relative_algorithm_difference_from_retained": float((direction - native_direction).norm()) / native_norm
                    if native_norm > 0 else None,
                "relative_displacement_difference_from_retained": float((displacement - native_displacement).norm()) / finite_norm
                    if finite_norm > 0 else None,
                "algorithm_cosine_with_retained": cosine(direction, native_direction),
                "next_minibatch_alignment": alignment(stochastic, directions), "populations": populations_after}
    return result
