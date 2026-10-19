"""Schedule-Free methods, arXiv:2405.15682v4, Sections 2 and 4.4.

SGD implements mix/query/average of Transformer.ScheduleFree.Section2_Averaging,
including the initial base point in the equal-weighted average. AdamW follows
Transformer.ScheduleFree.adamStep exactly, including epsilon placement, decay
at the query, warmup and squared effective-rate weights. These explicit-buffer
implementations store the query in parameters during training and evaluation
weights after eval(). They do not use the upstream package's algebraic buffer
compression or its extra weighting options.
"""

import torch

from ._parameters import Parameters, settings


class ScheduleFreeSGD(torch.optim.Optimizer):
    """Equal-weighted Schedule-Free SGD; beta may include both 0 and 1.

    Call optimizer.eval() for validation and optimizer.train() before the
    next training forward. The initial point has weight one.
    """
    def __init__(self, params, lr=1e-3, beta=0.9):
        if not 0 <= beta <= 1:
            raise ValueError("beta must lie in [0, 1]")
        settings(lr)
        model = Parameters(params)
        super().__init__(model.parameters(), dict(lr=lr, beta=beta, training=True))
        for p in self.param_groups[0]["params"]:
            self.state[p] = dict(base=p.detach().clone(), evaluation=p.detach().clone(), step=0)

    def add_param_group(self, group):
        if self.param_groups:
            raise ValueError("Schedule-Free methods support one parameter group")
        super().add_param_group(group)

    @torch.no_grad()
    def _mode(self, training):
        group = self.param_groups[0]
        group["training"] = training
        beta = group.get("beta", group.get("betas", (0.9,))[0])
        for p in group["params"]:
            state = self.state[p]
            p.copy_(beta * state["evaluation"] + (1 - beta) * state["base"]
                    if training else state["evaluation"])
        return self

    def train(self):
        return self._mode(True)

    def eval(self):
        return self._mode(False)

    def _gradients(self, closure):
        if not self.param_groups[0]["training"]:
            raise RuntimeError("Call optimizer.train() before a training forward/step")
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        parameters = self.param_groups[0]["params"]
        if any(p.grad is None or p.grad.is_sparse for p in parameters):
            raise ValueError("Schedule-Free methods require a dense gradient for every parameter")
        return loss, parameters

    @torch.no_grad()
    def step(self, closure=None):
        loss, parameters = self._gradients(closure)
        group = self.param_groups[0]
        for p in parameters:
            s = self.state[p]
            s["base"].add_(p.grad, alpha=-group["lr"])
            s["step"] += 1
            c = 1 / (s["step"] + 1)
            s["evaluation"].mul_(1 - c).add_(s["base"], alpha=c)
        self.train()
        return loss


class ScheduleFreeAdamW(ScheduleFreeSGD):
    """Section 4.4 Algorithm 1; Transformer.ScheduleFree.adamStep.

    eta_t = lr * sqrt(1-beta2**t) * min(1,t/warmup); epsilon is added to
    sqrt(raw_second), with decay evaluated at y. This is deliberately not
    the epsilon placement of torch.optim.AdamW. No nonconvex convergence
    theorem is claimed by the source or this package.
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), weight_decay=0.0, eps=1e-8, warmup=0):
        settings(lr, betas=betas, weight_decay=weight_decay, eps=eps)
        if type(warmup) is not int or warmup < 0:
            raise ValueError("warmup must be a nonnegative integer")
        super().__init__(params, lr, betas[0])
        self.param_groups[0].update(betas=betas, weight_decay=weight_decay, eps=eps, warmup=warmup)
        del self.param_groups[0]["beta"]
        for p in self.param_groups[0]["params"]:
            self.state[p].update(second=torch.zeros_like(p), total=0.0)

    @torch.no_grad()
    def step(self, closure=None):
        loss, parameters = self._gradients(closure)
        group = self.param_groups[0]
        beta2 = group["betas"][1]
        for p in parameters:
            s, g = self.state[p], p.grad
            s["step"] += 1
            warmup = min(1.0, s["step"] / group["warmup"]) if group["warmup"] else 1.0
            eta = group["lr"] * (1 - beta2 ** s["step"]) ** 0.5 * warmup
            s["second"].mul_(beta2).addcmul_(g, g, value=1 - beta2)
            s["total"] += eta ** 2
            c = eta ** 2 / s["total"] if s["total"] else 0.0
            # p is the old query y, needed for the source's independent decay.
            s["base"].sub_(eta * g / (s["second"].sqrt() + group["eps"]) + eta * group["weight_decay"] * p)
            s["evaluation"].mul_(1 - c).add_(s["base"], alpha=c)
        self.train()
        return loss
