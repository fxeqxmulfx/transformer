"""Tensor-counter versions of the frozen optimizer's numerical transitions."""

import torch

from experiments.optimizer_benchmark.coordinate import CoordinateOptimizer
from experiments.optimizer_benchmark.fisher import AdaFisherOptimizer, minmax
from experiments.optimizer_benchmark.matrix import DashOptimizer, MuonOptimizer
from experiments.magma_benchmark.optimizer import MagmaOptimizer, RMSPropOptimizer, cosine


def initialize(opt, counter):
    opt.steps = counter
    base = opt.base if isinstance(opt, MagmaOptimizer) else opt
    base.steps = counter
    for p in base.param_groups[0]["params"]:
        state = base.state[p]
        if isinstance(base, CoordinateOptimizer):
            state.update(m=torch.zeros_like(p), v=torch.zeros_like(p), maximum=torch.zeros_like(p))
        elif isinstance(base, MuonOptimizer):
            if p.ndim != 2 or base.names[p] == "embed.weight":
                state.update(m=torch.zeros_like(p), v=torch.zeros_like(p), maximum=torch.zeros_like(p))
            else:
                state["momentum"] = torch.zeros_like(p)
        elif isinstance(base, RMSPropOptimizer):
            state.update(m=torch.zeros_like(p), v=torch.zeros_like(p))
        elif isinstance(base, AdaFisherOptimizer):
            rows, columns = p.shape if p.ndim == 2 else (p.numel(), 1)
            state.update(h=p.new_zeros(columns), s=p.new_zeros(rows), moment=torch.zeros_like(p))
    if isinstance(base, DashOptimizer):
        offset, locations = 0, {}
        for p in base.param_groups[0]["params"]:
            locations[p] = torch.arange(p.numel(), device=p.device).reshape(
                p.shape if p.ndim == 2 else (p.numel(), 1)) + offset
            offset += p.numel()
        for group in base.groups:
            height, width = group["shape"]
            group["positions"] = torch.stack([locations[p][i:i+height, j:j+width]
                for p, i, j in group["entries"]])
            if base.solver == "chebyshev":
                # Warm the immutable cosine basis before Dynamo traces its use.
                base.roots(group["left"])
    if isinstance(opt, MagmaOptimizer):
        for p in opt.masked_parameters:
            state = opt.state[p]
            state.update(scale=p.new_tensor(.5), scale_sum=p.new_zeros(()),
                cosine_sum=p.new_zeros(()), draws=p.new_zeros((), dtype=torch.int64),
                kept=p.new_zeros((), dtype=torch.int64))
            if getattr(base, "rule", None) == "sgd":
                state["alignment_momentum"] = torch.zeros_like(p)


def coordinate(opt, parameters):
    step = opt.steps
    beta = opt.beta / step if opt.schedule == "inverse" else (
        opt.beta * .99 ** (step - 1) if opt.schedule == "geometric" else opt.beta)
    decrease = opt.schedule != "constant" or opt.rule in {"adagrad", "adamnc"}
    factor = step.rsqrt() if decrease else 1.
    result = []
    for p in parameters:
        g, state = p.grad, opt.state[p]
        m, v, maximum = state["m"], state["v"], state["maximum"]
        if opt.rule == "sgd":
            direction = g.clone()
        elif opt.rule in {"adagrad", "adamnc"}:
            v.mul_((step - 1) / step).add_(g.square() / step)
            if opt.rule == "adagrad":
                m.copy_(g)
            else:
                m.mul_(beta).add_(g * (1 - beta))
            direction = factor * m / (v.sqrt() + opt.eps)
        else:
            m.mul_(beta).add_(g * (1 - beta))
            v.mul_(opt.beta2).addcmul_(g, g, value=1 - opt.beta2)
            if opt.rule == "adamx":
                previous_beta = opt.beta / (step - 1).clamp_min(1)
                maximum.mul_(((1 - beta) / (1 - previous_beta)) ** 2)
            if opt.rule in {"adamx", "amsgrad"}:
                maximum.copy_(torch.maximum(maximum, v))
                denominator = maximum.sqrt() + opt.eps
            elif opt.rule == "adamw":
                denominator = (v / (1 - opt.beta2 ** step)).sqrt() + opt.eps
            else:
                denominator = v.sqrt() + opt.eps
            numerator = m / (1 - beta ** step) if opt.rule == "adamw" else m
            direction = factor * numerator / denominator
        result.append(direction + opt.decay * p if opt.decay else direction)
    return result


def dash(opt, parameters):
    packed = torch.cat([p.grad.reshape(-1) for p in parameters])
    directions = torch.zeros_like(packed)
    for group in opt.groups:
        gradient = packed[group["positions"]]
        transpose = gradient.transpose(-2, -1)
        group["left"].mul_(opt.beta).add_(gradient @ transpose, alpha=1 - opt.beta)
        group["right"].mul_(opt.beta).add_(transpose @ gradient, alpha=1 - opt.beta)
        group["moment"].mul_(opt.graft_beta).add_(gradient, alpha=1 - opt.graft_beta)
        group["accumulator"].mul_(opt.graft_beta2).addcmul_(gradient, gradient, value=1 - opt.graft_beta2)
        reference = group["moment"] / (group["accumulator"].sqrt() + opt.epsilon)
        output = opt.roots(group["left"]) @ gradient @ opt.roots(group["right"])
        norm = output.square().sum((-2, -1)).sqrt()
        ref_norm = reference.square().sum((-2, -1)).sqrt()
        output = output * (ref_norm / torch.where(norm > 0, norm, torch.ones_like(norm)))[..., None, None]
        directions.scatter_(0, group["positions"].flatten(), output.flatten())
    offset, result = 0, []
    for p in parameters:
        result.append(directions[offset:offset+p.numel()].reshape_as(p))
        offset += p.numel()
    return result


def propose(opt, parameters, factors, masks):
    if isinstance(opt, MagmaOptimizer):
        result, index = [], 0
        for p, direction in zip(parameters, propose(opt.base, parameters, factors, masks)):
            if p not in opt.masked_parameters:
                result.append(direction)
                continue
            state = opt.state[p]
            dense = opt.base.state[p]
            if "momentum" in dense:
                moment = dense["momentum"]
            elif getattr(opt.base, "rule", None) != "sgd" and "m" in dense:
                moment = dense["m"]
            else:
                moment = state["alignment_momentum"].mul_(.9).add_(p.grad, alpha=.1)
            alignment = cosine(moment, p.grad)
            state["scale"].mul_(.9).add_(torch.sigmoid(alignment / opt.tau), alpha=.1)
            state["draws"].add_(1)
            state["kept"].add_(masks[index])
            state["scale_sum"].add_(state["scale"])
            state["cosine_sum"].add_(alignment)
            result.append(torch.where(masks[index], direction * state["scale"], torch.zeros_like(direction)))
            index += 1
        return result
    if isinstance(opt, CoordinateOptimizer):
        return coordinate(opt, parameters)
    if isinstance(opt, DashOptimizer):
        return dash(opt, parameters)
    if isinstance(opt, AdaFisherOptimizer):
        result = []
        for p in parameters:
            rows, columns = p.shape if p.ndim == 2 else (p.numel(), 1)
            h, s = factors.get(p, (p.new_ones(columns), p.new_ones(rows)))
            state = opt.state[p]
            state["h"].mul_(1-opt.gamma).add_(h, alpha=opt.gamma)
            state["s"].mul_(1-opt.gamma).add_(s, alpha=opt.gamma)
            state["moment"].mul_(opt.beta).add_(p.grad, alpha=1-opt.beta)
            metric = (minmax(state["s"])[:, None] * minmax(state["h"])[None, :] + opt.damping).reshape_as(p)
            direction = (state["moment"] / (1-opt.beta ** opt.steps)) / metric
            result.append(direction + opt.decay*p if opt.decay else direction)
        return result
    return opt.propose(parameters)
