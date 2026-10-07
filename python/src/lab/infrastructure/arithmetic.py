"""Audited ATen reference-arithmetic counts for controlled Basis comparisons.

Version 1: one multiply/add is two FLOPs; each scalar arithmetic,
comparison, selection and transcendental is one reference operation.
Integer/Boolean arithmetic is recorded separately and charged one unit
when it is auxiliary data work. Copies, reshapes, indexing and allocation
are zero arithmetic. These are shape-based ATen reference counts, not
hardware instruction counters or a prediction of wall time. Stable
log-sum-exp uses a maximum, shifted exponentials, sum, log and addition;
log-prefix sums use an eight-operation stable logaddexp recurrence.
Every encountered operator must have an explicit rule. Unsupported ones
raise, rather than silently contributing zero as PyTorch's default does.
"""

from collections import Counter
import math

import torch
from torch.utils._python_dispatch import TorchDispatchMode

VERSION = "aten-reference-arithmetic-v1"

MOVEMENT = {
    "_unsafe_view", "alias", "arange", "as_strided", "cat", "clone", "constant_pad_nd", "contiguous", "copy_",
    "detach", "embedding", "empty", "empty_like", "empty_strided", "expand", "fill_", "flip", "full", "full_like",
    "gather", "index", "index_select", "lift_fresh", "lift_fresh_copy", "new_empty",
    "new_empty_strided", "new_zeros", "ones", "ones_like", "permute", "reshape", "scalar_tensor", "select",
    "select_backward", "slice", "slice_backward", "split", "split_with_sizes", "squeeze", "stack", "t", "to",
    "_to_copy", "transpose", "triu", "tril", "unbind", "unsqueeze", "view", "view_as", "zero_", "zeros", "zeros_like",
}
ELEMENTWISE = {
    "abs", "add", "add_", "bitwise_and", "bitwise_and_", "bitwise_not", "bitwise_or", "clamp", "clamp_",
    "clamp_min", "clamp_min_", "div", "div_", "eq", "exp", "exp_", "floor_divide", "ge", "gt", "isfinite",
    "le", "log", "log1p", "logical_and", "logical_not", "lt", "masked_fill", "masked_fill_", "maximum", "minimum", "mul", "mul_", "ne",
    "neg", "pow", "reciprocal", "relu", "remainder", "rsub", "rsqrt", "sigmoid", "sin", "cos", "sqrt", "sub", "sub_",
    "where",
}


def tensors(value):
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (list, tuple)):
        return [tensor for item in value for tensor in tensors(item)]
    return []


def count_rule(name, args, kwargs, result):
    """Return scalar reference operations and the tensor deciding their dtype."""
    outputs, inputs = tensors(result), tensors(args)
    first = inputs[0] if inputs else (outputs[0] if outputs else None)
    output = outputs[0] if outputs else first
    n = 0 if output is None else output.numel()
    if name in MOVEMENT or name.startswith("profiler::"):
        return 0, output
    if name in ELEMENTWISE:
        if name in {"clamp", "clamp_"}:
            lower = args[1] if len(args) > 1 else kwargs.get("min")
            upper = args[2] if len(args) > 2 else kwargs.get("max")
            return n * ((lower is not None) + (upper is not None)), output
        return n, first if name in {"eq", "ge", "gt", "le", "lt", "ne", "isfinite"} else output
    if name in {"mm", "bmm"}:
        return 2 * n * args[0].shape[-1], output
    if name == "addmm":
        return 2 * n * args[1].shape[-1] + n, output
    if name in {"sum", "mean", "amax", "amin", "max", "min", "argmax", "argmin"}:
        return max(0, first.numel() - n) + (n if name == "mean" else 0), first
    if name in {"cumsum", "cumprod"}:
        dimension = args[1] % first.ndim
        return first.numel() - first.numel() // first.shape[dimension], first
    if name == "logcumsumexp":
        dimension = args[1] % first.ndim
        return 8 * (first.numel() - first.numel() // first.shape[dimension]), first
    if name == "logsumexp":
        return 4 * first.numel(), first
    if name in {"_softmax", "_log_softmax"}:
        rows = first.numel() // first.shape[args[1]]
        return 5 * first.numel() - (2 if name == "_softmax" else 1) * rows, first
    if name in {"_softmax_backward_data", "_log_softmax_backward_data"}:
        rows = first.numel() // first.shape[args[2]]
        return 4 * first.numel() - rows, first
    if name == "linalg_vector_norm":
        if args[1] != 2:
            raise NotImplementedError("Only the actual L2 gradient norm is counted")
        return 2 * first.numel(), first
    if name == "embedding_dense_backward":
        return first.numel(), first
    if name in {"scatter_add", "scatter_add_", "index_add", "index_add_"}:
        return args[3].numel(), args[3]
    if name in {"index_put", "index_put_", "_index_put_impl_"}:
        accumulate = args[3] if len(args) > 3 else kwargs.get("accumulate", False)
        return args[2].numel() if accumulate else 0, args[2]
    if name == "threshold_backward":
        return 2 * first.numel(), first
    if name == "nll_loss_forward":
        return 2 * args[1].numel() + 1, first
    if name == "nll_loss_backward":
        return args[2].numel() + 1, output
    if name in {"sort", "argsort"}:
        dimension = kwargs.get("dim", args[1] if len(args) > 1 and type(args[1]) is int else -1)
        # Explicit reference comparison charge; ordering/memory traffic
        # is not a hardware FLOP measure and differs across sorting kernels.
        return first.numel() * math.ceil(math.log2(max(1, first.shape[dimension]))), first
    if name == "_foreach_norm":
        return sum(2 * item.numel() for item in args[0]), args[0][0]
    if name.startswith(("_foreach_add", "_foreach_mul", "_foreach_div", "_foreach_sub")):
        return sum(item.numel() for item in args[0]), args[0][0]
    if name == "_fused_adamw_":
        # Standard AdamW with decay, two moments and bias corrections:
        # 14 operations per coordinate and nine scalar operations per
        # parameter tensor. Kernel fusion/reassociation is not reweighted.
        return 14 * sum(item.numel() for item in args[0]) + 9 * len(args[0]), args[0][0]
    raise NotImplementedError(f"Uncounted arithmetic operator: {name}")


class Arithmetic(TorchDispatchMode):
    """Observe executed operator shapes, with complete rule coverage."""
    def __init__(self):
        self.floating = self.integer = 0
        self.operators = Counter()
        self.calls = Counter()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        result = func(*args, **kwargs)
        schema = func._schema.name
        name = schema if schema.startswith("profiler::") else schema.split("::")[-1]
        operations, dtype_source = count_rule(name, args, kwargs, result)
        floating = dtype_source is not None and (dtype_source.is_floating_point() or dtype_source.is_complex())
        if floating:
            self.floating += operations
        else:
            self.integer += operations
        self.operators[str(func)] += operations
        self.calls[str(func)] += 1
        return result

    @property
    def charged(self):
        return self.floating + self.integer

    def report(self):
        return {"convention": VERSION, "floating_ops": self.floating, "integer_ops": self.integer,
                "charged_ops": self.charged, "operators": dict(sorted(self.operators.items())),
                "calls": dict(sorted(self.calls.items()))}
