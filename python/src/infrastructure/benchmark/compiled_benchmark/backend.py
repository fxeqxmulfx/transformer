"""Inductor/CUDA Graph adapter for the frozen GPTMini model.

Finite causal Sparsemax rows use the identical centered projection and support
Jacobian. Invalid rows signal NaN to the outside-graph nonfinite loss check.
Compile the module in place so optimizer names, objects and tied weights stay
intact. Each forward explicitly begins a CUDA graph iteration.
"""

import torch
from torch.nn import functional as F

from gpt_mini.gpt_mini import CausalMHA, apply_rope


class GraphSparsemaxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores):
        maximum = scores.amax(dim=-1, keepdim=True)
        finite = maximum.isfinite()
        centered = scores - torch.where(finite, maximum, torch.zeros_like(maximum))
        ordered = centered.sort(dim=-1, descending=True).values
        cumulative = ordered.cumsum(dim=-1)
        ranks = torch.arange(1, scores.size(-1) + 1, device=scores.device, dtype=scores.dtype)
        size = (1 + ranks * ordered > cumulative).sum(dim=-1, keepdim=True).clamp_min(1)
        threshold = (cumulative.gather(-1, size - 1) - 1) / size
        probabilities = (centered - threshold).clamp_min(0)
        probabilities = torch.where(finite, probabilities, torch.full_like(probabilities, float("nan")))
        active = probabilities > 0
        ctx.save_for_backward(active)
        return probabilities

    @staticmethod
    def backward(ctx, gradient):
        (active,) = ctx.saved_tensors
        count = active.sum(dim=-1, keepdim=True).clamp_min(1)
        mean = (gradient * active).sum(dim=-1, keepdim=True) / count
        return active * (gradient - mean)


def graph_sparsemax(scores):
    return GraphSparsemaxFunction.apply(scores)


class GraphSparsemaxMHA(CausalMHA):
    def forward(self, x, cos, sin):
        batch, length, width = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        q = apply_rope(F.normalize(q, dim=-1, eps=1e-6), cos, sin)
        k = apply_rope(F.normalize(k, dim=-1, eps=1e-6), cos, sin)
        alpha = self.log_alpha.exp().view(1, self.n_heads, 1, 1)
        scores = (q @ k.transpose(-2, -1)) * alpha
        forbidden = torch.ones((length, length), device=x.device, dtype=torch.bool).triu(1)
        probabilities = graph_sparsemax(scores.masked_fill(forbidden, float("-inf")))
        y = probabilities @ v
        v_hat = F.normalize(v, dim=-1, eps=1e-6)
        z = y - (y * v_hat).sum(dim=-1, keepdim=True) * v_hat
        return self.proj(z.transpose(1, 2).contiguous().view(batch, length, width))


def compile_model(model, attention, method, *, mode="reduce-overhead"):
    if attention == "sparsemax":
        for block in model.blocks:
            block.attn.__class__ = GraphSparsemaxMHA
    fullgraph = not method.startswith("adafisher")
    model.compile(backend="inductor", mode=mode, fullgraph=fullgraph,
                  dynamic=False, isolate_recompiles=True)
    compiled_call = model._compiled_call_impl

    def marked_call(*args, **kwargs):
        torch.compiler.cudagraph_mark_step_begin()
        return compiled_call(*args, **kwargs)

    model._compiled_call_impl = marked_call
    model.compile_settings = {"backend": "inductor", "mode": mode, "fullgraph": fullgraph,
                              "dynamic": False, "step_marked": True}
    return model


def graph_counts():
    # Runtime evidence for installed Torch 2.14; this is diagnostic, not part
    # of optimizer mathematics. Count actual CUDAGraph objects, recursively.
    from torch._inductor.cudagraph_trees import get_manager
    manager = get_manager(create_if_none_exists=False)
    if manager is None:
        return {"recorded_graph_nodes": 0}
    seen = set()

    def visit(node):
        if id(node) in seen:
            return 0
        seen.add(id(node))
        count = int(node.graph is not None)
        for children in node.children.values():
            count += sum(visit(child) for child in children)
        return count

    return {"recorded_graph_nodes": sum(visit(root) for root in manager.get_roots())}
