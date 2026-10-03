"""Use the frozen stopping loop with a scoped in-place compiler factory.

The parent experiment is already measured. Its source files stay unchanged;
only this solo process's factory/evaluation bindings are overridden, and both
are restored even on interruption. Optimizers and stopping decisions use the
exact parent implementation. Compiler state is reset between models while the
disk cache remains available.
"""

import gc
import time

import torch

from experiments.patience_benchmark import training as parent
from .backend import compile_model
from .telemetry import snapshot


def train_one(config, data, attention, method, rate, seed, batch, stopping, artifacts,
              *, device="cuda", progress=None):
    if torch.device(device).type != "cuda":
        raise ValueError("The compiled experiment requires CUDA")
    torch.compiler.reset()
    from torch._dynamo.utils import counters
    counters.clear()
    original_factory, original_evaluate = parent.make_model, parent.evaluate
    details = {"cold_forward_seconds": {}, "evaluations": []}

    def factory(cfg, attn, model_seed, target):
        model = compile_model(original_factory(cfg, attn, model_seed, target), attn, method)
        details.update(model.compile_settings)
        call = model._compiled_call_impl

        def first_call_timing(*args, **kwargs):
            phase = "train" if torch.is_grad_enabled() else "inference"
            if phase in details["cold_forward_seconds"]:
                return call(*args, **kwargs)
            torch.cuda.synchronize()
            started = time.perf_counter()
            result = call(*args, **kwargs)
            torch.cuda.synchronize()
            details["cold_forward_seconds"][phase] = time.perf_counter() - started
            return result

        model._compiled_call_impl = first_call_timing
        return model

    def evaluation(model, tokens, context, eval_batch):
        started = time.perf_counter()
        value = original_evaluate(model, tokens, context, eval_batch)
        details["evaluations"].append({"split": "test" if tokens is data.test else "validation",
            "seconds": time.perf_counter() - started, **snapshot()})
        if progress is not None:
            current = details["evaluations"][-1]
            progress(f"compiler {attention} {method} seed={seed} "
                     f"FX={current['unique_fx_graphs']} CUDA_nodes={current['recorded_graph_nodes']}")
        return value

    parent.make_model, parent.evaluate = factory, evaluation
    try:
        row = parent.train_one(config, data, attention, method, rate, seed, batch, stopping,
                               artifacts, device=device, progress=progress)
        details["final"] = snapshot()
        details["peak_reserved_mib"] = torch.cuda.max_memory_reserved() / 2**20
        row["compile"] = details
        return row
    finally:
        parent.make_model, parent.evaluate = original_factory, original_evaluate
        torch.compiler.reset()
        gc.collect()
        torch.cuda.empty_cache()
