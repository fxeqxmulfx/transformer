"""Capture every Execution.ModelTrace boundary of the pinned source forward.

Source: review/flat_spectrum.md, Complete captured execution, L1. The actual
GPT.forward executes under observational hooks; no branch is replaced by an
equation-based surrogate. Returning intermediates can change compiler fusion,
so this certifies the instrumented compiled invocation. SDPA scores are an
explicit auxiliary dot product: its average is checked at those scores, not
asserted to expose a fused kernel's hidden state. No training is performed.
"""

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import platform
import sys

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


ROOT = Path(__file__).resolve().parent


class Observed(nn.Module):
    """Run the unchanged source forward while retaining its operation tensors."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        block = model.blocks[0]
        for name, module in (("fused", block.sa.c_attn),
                             ("projected", block.sa.c_proj),
                             ("gate_value", block.ffwd.c_fc),
                             ("ffn_projected", block.ffwd.c_proj),
                             ("output", block)):
            module.register_forward_hook(self.output_hook(name))
        block.sa.c_proj.register_forward_pre_hook(self.input_hook("heads"))
        block.ln_2.register_forward_pre_hook(self.input_hook("mid"))
        block.ffwd.c_proj.register_forward_pre_hook(self.input_hook("product"))

    def output_hook(self, name):
        def hook(module, args, output):
            self.data[name] = output
        return hook

    def input_hook(self, name):
        def hook(module, args):
            self.data[name] = args[0]
        return hook

    def forward(self, tokens):
        self.data = {}
        self.norms = []
        self.rotations = []
        logits = self.model(tokens)[0]
        initial, pre, qnorm, knorm, attn_out, ffn_pre, ffn_out, normalized = self.norms
        gate, value = self.data.pop("gate_value").chunk(2, -1)
        self.data.update(initial=initial, pre=pre, qnorm=qnorm, knorm=knorm,
                         attn_output=attn_out, ffn_pre=ffn_pre, ffn_output=ffn_out,
                         normalized=normalized, gate=gate, value=value,
                         query=self.rotations[0], key=self.rotations[1], logits=logits)
        return self.data


@contextmanager
def observe(observed, mha):
    """Observational wrappers preserve each original functional operation."""
    rms, normalize, silu, sdpa, rope = F.rms_norm, F.normalize, F.silu, F.scaled_dot_product_attention, mha.apply_rope

    def norm_call(*args, **kwargs):
        result = rms(*args, **kwargs)
        if torch.compiler.is_dynamo_compiling():
            observed.norms.append(result)
        return result

    def direction_call(*args, **kwargs):
        result = normalize(*args, **kwargs)
        if torch.compiler.is_dynamo_compiling():
            observed.data["direction"] = result
        return result

    def silu_call(*args, **kwargs):
        result = silu(*args, **kwargs)
        if torch.compiler.is_dynamo_compiling():
            observed.data["activated"] = result
        return result

    def average_call(q, k, v, **kwargs):
        if torch.compiler.is_dynamo_compiling():
            observed.data["scores"] = q @ k.transpose(-1, -2) / 2
        result = sdpa(q, k, v, **kwargs)
        if torch.compiler.is_dynamo_compiling():
            observed.data["average"] = result
        return result

    def rope_call(*args, **kwargs):
        result = rope(*args, **kwargs)
        if torch.compiler.is_dynamo_compiling():
            observed.rotations.append(result)
        return result

    F.rms_norm, F.normalize, F.silu, F.scaled_dot_product_attention, mha.apply_rope = norm_call, direction_call, silu_call, average_call, rope_call
    try:
        yield
    finally:
        F.rms_norm, F.normalize, F.silu, F.scaled_dot_product_attention, mha.apply_rope = rms, normalize, silu, sdpa, rope


def parameters(model):
    """Fixed nonzero dyadic weights; all attention and SwiGLU matrices are active."""
    with torch.no_grad():
        model.token_embedding_table.weight.copy_(torch.eye(3, 4))
        for name, parameter in model.named_parameters():
            if name == "token_embedding_table.weight":
                continue
            rows, columns = parameter.shape
            values = [[((3 * r + 2 * c + 1) % 7 - 3) for c in range(columns)]
                      for r in range(rows)]
            matrix = torch.tensor(values, dtype=torch.float32)
            if name.endswith("sa.c_attn.weight"):
                matrix[:8] /= 65536
                matrix[8:] /= 8
            else:
                matrix /= 32
            parameter.copy_(matrix)


def main():
    source = json.loads((ROOT / "sequence_model_source.json").read_text())
    for name, digest in source["files"].items():
        if hashlib.sha256((Path(source["source_root"]) / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"pinned source changed: {name}")
    sys.path.insert(0, str(Path(source["source_root"]).parents[2]))
    from diff_gpt.domain.model.gpt import GPT
    from diff_gpt.domain.model import mha

    torch.set_num_threads(1)
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    torch.backends.cudnn.conv.fp32_precision = "ieee"
    model = GPT(3, 4, 2, 1, 1, use_checkpoint=False, pad_vocab_size_to=1).eval()
    parameters(model)
    model = model.cuda()
    tokens = torch.tensor([[0, 1]], device="cuda")
    with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
        plain = torch.compile(model, mode="max-autotune", fullgraph=True, dynamic=False)(tokens)[0].clone()
        observed = Observed(model)
        with observe(observed, mha):
            compiled = torch.compile(observed, mode="max-autotune", fullgraph=True, dynamic=False)
            compiled(tokens)
            torch.cuda.synchronize()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                    torch.profiler.ProfilerActivity.CUDA]) as profile:
                tensors = compiled(tokens)
                torch.cuda.synchronize()
    record = lambda tensor: {"shape": list(tensor.shape), "values": tensor.detach().cpu().flatten().tolist()}
    report = {
        "source_commit": source["repository_commit"], "source_hashes": source["files"],
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python": platform.python_version(), "torch": torch.__version__,
        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
        "compile": "max-autotune/fullgraph/static", "dtype": "torch.float32", "sdpa_backend": "MATH",
        "config": {"width": 4, "heads": 1, "depth": 1, "ffn": 11, "vocab": 3, "length": 2, "rope_base": 10000},
        "tokens": [0, 1], "rms_epsilon": "1/8388608", "l2_epsilon": "1/1000000000000",
        "parameters": {name: record(tensor) for name, tensor in model.named_parameters()},
        "tables": {"cos": record(model.rope_cos), "sin": record(model.rope_sin)},
        "trace": {name: record(tensor) for name, tensor in tensors.items()},
        "uninstrumented_logits": record(plain),
        "instrumentation_max_logit_difference": float((plain - tensors["logits"]).abs().max()),
        "observed_events": sorted({event.key for event in profile.key_averages()}),
        "scope": "One complete instrumented compiled source invocation; no uniform kernel or training guarantee.",
    }
    path = ROOT / "model_trace_capture.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"report": str(path), "sites": len(tensors), "logits": report["trace"]["logits"],
                      "instrumentation_max_logit_difference": report["instrumentation_max_logit_difference"]}, indent=2))


if __name__ == "__main__":
    main()
