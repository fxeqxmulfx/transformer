"""Compile loss, reverse AD, optimizer states and parameter updates together."""

import math
import torch
from torch.nn import functional as F

from experiments.optimizer_benchmark.data import windows
from experiments.patience_benchmark.registry import make_optimizer
from .model import forward, linear_names, output_shapes
from .optimizer import initialize, propose


def mask_plan(steps, count, seed):
    rng = torch.Generator(device="cpu").manual_seed(20000 + seed)
    return torch.rand((steps, count), generator=rng) < .5


class FullStep:
    def __init__(self, model, attention, method, rate, seed, batch, max_steps, *, compiled=True):
        self.model, self.attention, self.method = model, attention, method
        self.parameters = dict(model.named_parameters())
        self.opt = make_optimizer(method, model, rate, seed)
        if hasattr(self.opt, "masked_parameters"):
            self.opt.masked_parameters = dict.fromkeys(self.opt.masked_parameters)
        self.counter = model.embed.weight.new_zeros((), dtype=torch.float64)
        self.accepted = model.embed.weight.new_zeros((), dtype=torch.int64)
        self.eval_total = model.embed.weight.new_zeros(())
        initialize(self.opt, self.counter)
        for p in self.parameters.values():
            p.grad = torch.zeros_like(p)
        self.fisher = method.startswith("adafisher")
        self.factors = {}
        self.probes = ()
        if self.fisher:
            self.opt.collector.close()
            self.probes = tuple(model.embed.weight.new_zeros(batch, model.cfg.max_seq_len, width)
                                for width in output_shapes(model.cfg))
            for name in linear_names(model.cfg):
                p = self.parameters[name]
                self.factors[p] = (p.new_zeros(p.shape[1]), p.new_zeros(p.shape[0]))
        masked = len(self.opt.masked_parameters) if hasattr(self.opt, "masked_parameters") else 0
        self.masks = mask_plan(max_steps, masked, seed).to(model.embed.weight.device)
        # These tensors are updated in place and never replaced. Mark their
        # real stable addresses so CUDA Graphs do not copy a mutable state
        # input into disposable staging storage or skip capture.
        def mark(value):
            if isinstance(value, torch.Tensor):
                torch._dynamo.mark_static_address(value, guard=True)
            elif isinstance(value, dict):
                for child in value.values():
                    mark(child)
            elif isinstance(value, (tuple, list)):
                for child in value:
                    mark(child)

        mark((self.parameters, [p.grad for p in self.parameters.values()], self.counter,
              self.accepted, self.eval_total, self.masks, self.probes, self.factors, model.cos, model.sin))
        mark(self.opt.state)
        base = self.opt.base if hasattr(self.opt, "base") else self.opt
        mark(base.state)
        mark(getattr(base, "groups", ()))
        if self.fisher:
            self.gradient = torch.func.grad_and_value(self._fisher_loss, argnums=(0, 1), has_aux=True)
        else:
            self.gradient = torch.func.grad_and_value(self._loss)
        self.compilation_settings = {"backend": "inductor", "mode": "reduce-overhead",
            "fullgraph": True, "dynamic": False, "model_loss_gradient_optimizer_compiled": True,
            "windows_and_evaluation_compiled": True, "tensor_counter": True,
            "magma_mask_stream": "preserved precomputed CPU plan, indexed inside GPU graph"}
        compile_fn = lambda fn: torch.compile(fn, backend="inductor", mode="reduce-overhead",
                fullgraph=True, dynamic=False, isolate_recompiles=True) if compiled else fn
        self.batch_kernel = compile_fn(self._batch)
        self.train_kernel = compile_fn(self._train)
        self.eval_kernel = compile_fn(self._evaluate)

    def _loss(self, parameters, inputs, targets):
        logits, _ = forward(self.model.cfg, self.attention, parameters,
                            self.model.cos, self.model.sin, inputs)
        return F.cross_entropy(logits.flatten(0, 1), targets.flatten())

    def _fisher_loss(self, parameters, probes, inputs, targets):
        logits, h = forward(self.model.cfg, self.attention, parameters,
                            self.model.cos, self.model.sin, inputs, probes)
        return F.cross_entropy(logits.flatten(0, 1), targets.flatten()), h

    def _batch(self, inputs, targets):
        if self.fisher:
            (gradients, derivatives), (loss, h) = self.gradient(self.parameters, self.probes, inputs, targets)
        else:
            gradients, loss = self.gradient(self.parameters, inputs, targets)
        with torch.no_grad():
            if self.fisher:
                for name, fresh_h, derivative in zip(linear_names(self.model.cfg), h, derivatives):
                    p = self.parameters[name]
                    fresh_s = derivative.detach().flatten(0, 1).square().sum(0)
                    self.factors[p][0].copy_(fresh_h)
                    self.factors[p][1].copy_(fresh_s)
            for name, p in self.parameters.items():
                p.grad.copy_(gradients[name])
            masks = self.masks.index_select(0, self.counter.long().reshape(1)).squeeze(0)
            self.counter.add_(1)
            parameters = list(self.parameters.values())
            directions = propose(self.opt, parameters, self.factors, masks)
            if self.opt.guarded:
                norm_g = sum(p.grad.square().sum() for p in parameters)
                norm_d = sum(d.square().sum() for d in directions)
                alignment = sum((p.grad * d).sum() for p, d in zip(parameters, directions))
                accepted = (alignment >= self.opt.sigma * norm_g) & (norm_d <= norm_g)
                self.accepted.add_(accepted)
                directions = [torch.where(accepted, d, p.grad) for p, d in zip(parameters, directions)]
            finite = loss.isfinite()
            for p, direction in zip(parameters, directions):
                p.add_(torch.where(finite, direction, torch.zeros_like(direction)), alpha=-self.opt.param_groups[0]["lr"])
        return loss.detach()

    def _train(self, tokens, starts):
        row = starts.index_select(0, self.counter.long().reshape(1)).squeeze(0)
        inputs, targets = windows(tokens, row, self.model.cfg.max_seq_len)
        return self._batch(inputs, targets)

    def _evaluate(self, tokens, starts, reset):
        with torch.no_grad():
            if reset:
                self.eval_total.zero_()
            inputs, targets = windows(tokens, starts, self.model.cfg.max_seq_len)
            logits, _ = forward(self.model.cfg, self.attention, self.parameters,
                                self.model.cos, self.model.sin, inputs)
            self.eval_total.add_(F.cross_entropy(logits.flatten(0, 1), targets.flatten(), reduction="sum"))
        return self.eval_total

    def batch(self, inputs, targets):
        torch.compiler.cudagraph_mark_step_begin()
        return self.batch_kernel(inputs, targets)

    def train(self, tokens, starts):
        torch._dynamo.mark_static_address(tokens, guard=True)
        torch._dynamo.mark_static_address(starts, guard=True)
        torch.compiler.cudagraph_mark_step_begin()
        return self.train_kernel(tokens, starts)

    def evaluate(self, tokens, batch):
        starts = torch.arange(0, len(tokens)-self.model.cfg.max_seq_len, self.model.cfg.max_seq_len).to(tokens.device)
        for offset in range(0, len(starts), batch):
            torch.compiler.cudagraph_mark_step_begin()
            self.eval_kernel(tokens, starts[offset:offset+batch], offset == 0)
        value = self.eval_total.item() / (len(starts) * self.model.cfg.max_seq_len)
        if not math.isfinite(value):
            raise FloatingPointError("Nonfinite compiled held-out loss")
        return value

    def assert_states_match(self, reference, *, rtol, atol):
        def compare(a, b):
            if isinstance(a, dict):
                for key in a:
                    compare(a[key], b[key])
            elif isinstance(a, torch.Tensor):
                torch.testing.assert_close(a, b, rtol=rtol, atol=atol)
            elif isinstance(b, torch.Tensor):
                assert a == b.item(), (a, b)
            else:
                assert a == b, (a, b)

        def optimizer_states(a, b):
            for p, q in zip(a.param_groups[0]["params"], b.param_groups[0]["params"]):
                compare(a.state[p], b.state[q])
            if hasattr(a, "base"):
                optimizer_states(a.base, b.base)
            if hasattr(a, "groups"):
                for first, second in zip(a.groups, b.groups):
                    for key in ("left", "right", "moment", "accumulator"):
                        compare(first[key], second[key])

        optimizer_states(reference, self.opt)
        assert reference.steps == self.counter.item()
        assert reference.guard_accepted == self.accepted.item()

    def close(self):
        self.opt.close()
