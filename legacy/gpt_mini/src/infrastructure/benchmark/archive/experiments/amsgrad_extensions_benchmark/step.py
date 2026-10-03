"""Reuse the frozen functional model and replace only the optimizer recurrence."""

import torch

from experiments.full_compile_benchmark.step import FullStep
from .optimizer import ExtensionOptimizer


class ExtensionStep(FullStep):
    def __init__(self, model, attention, method, rate, seed, batch, max_steps, *, compiled=True):
        # The parent provides loss/reverse AD, window gathering, held-out
        # reductions and stable parameter addresses. No frozen source is edited.
        super().__init__(model, attention, "amsgrad", rate, seed, batch, max_steps, compiled=False)
        self.opt.close()
        self.method = method
        self.opt = ExtensionOptimizer(self.parameters, method, rate,
                                      counter=self.counter, accepted=self.accepted)

        def mark(value):
            if isinstance(value, torch.Tensor):
                torch._dynamo.mark_static_address(value, guard=True)
            elif isinstance(value, dict):
                for child in value.values():
                    mark(child)
        mark(self.opt.state)
        mark(self.opt.halved)
        self.compilation_settings = {"backend": "inductor", "mode": "reduce-overhead",
            "fullgraph": True, "dynamic": False, "model_loss_gradient_optimizer_compiled": True,
            "windows_and_evaluation_compiled": True, "tensor_counter": True,
            "factor_gradients_projection_guard_repair_compiled": True}
        compile_fn = lambda fn: torch.compile(fn, backend="inductor", mode="reduce-overhead",
            fullgraph=True, dynamic=False, isolate_recompiles=True) if compiled else fn
        self.batch_kernel = compile_fn(self._batch)
        self.train_kernel = compile_fn(self._train)
        self.eval_kernel = compile_fn(self._evaluate)

    def _batch(self, inputs, targets):
        gradients, loss = self.gradient(self.parameters, inputs, targets)
        with torch.no_grad():
            for name, p in self.parameters.items():
                p.grad.copy_(gradients[name])
            self.opt.step(gradients, loss.isfinite())
        return loss.detach()
