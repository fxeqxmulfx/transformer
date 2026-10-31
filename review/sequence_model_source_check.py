"""Check G4's explicit equations against the user's pinned diff_gpt forward.

This is a source-correspondence diagnostic, not a learning experiment or Lean
extraction. Source hashes prevent silently validating a different architecture.
CPU float64 compares logits and every trainable gradient; CUDA float32 also
checks the original forward compiled with max-autotune/fullgraph/static shapes.
Run through `uv run --locked` inside the repository's python directory.
"""

import hashlib
import json
from pathlib import Path
import platform
import sys

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
SOURCE = json.loads((ROOT / "sequence_model_source.json").read_text())


def equations(model, tokens):
    """Literal scalar normalization, causal sums and source matrix coordinates."""
    eps = torch.finfo(model.token_embedding_table.weight.dtype).eps

    def rms(x):
        return x / (x.square().mean(-1, keepdim=True) + eps).sqrt()

    def rotary(x):
        first, second = x.chunk(2, -1)
        cos, sin = model.rope_cos[:tokens.shape[1]], model.rope_sin[:tokens.shape[1]]
        return torch.cat((first * cos - second * sin, first * sin + second * cos), -1)

    embedding = model.token_embedding_table.weight
    x = rms(embedding[tokens])
    for block in model.blocks:
        sa = block.sa
        q, k, v = (rms(x) @ sa.c_attn.weight.T).chunk(3, -1)
        split = lambda z: z.reshape(*z.shape[:2], sa.n_head, sa.head_dim).transpose(1, 2)
        q, k, v = split(q), split(k), split(v)
        q, k = rms(q), rms(k)
        if sa.use_rope:
            q, k = rotary(q), rotary(k)
        scores = q @ k.transpose(-1, -2) / sa.head_dim**.5
        future = torch.ones_like(scores, dtype=torch.bool).triu(1)
        weights = scores.masked_fill(future, -torch.inf).softmax(-1)
        y = weights @ v
        own = v / v.square().sum(-1, keepdim=True).sqrt().clamp_min(1e-12)
        y = y - (y * own).sum(-1, keepdim=True) * own
        y = torch.cat((torch.zeros_like(y[:, :, :1]), y[:, :, 1:]), 2)
        y = y.transpose(1, 2).reshape_as(x) @ sa.c_proj.weight.T
        x = x + rms(y)
        gate, value = (rms(x) @ block.ffwd.c_fc.weight.T).chunk(2, -1)
        branch = (gate / (1 + (-gate).exp()) * value) @ block.ffwd.c_proj.weight.T
        x = x + rms(branch)
    return rms(x) @ embedding.T


def compare(model, tokens, targets, forward):
    parameters = tuple(model.parameters())
    actual = forward(tokens)[0]
    expected = equations(model, tokens)
    actual_loss = F.cross_entropy(actual.flatten(0, 1), targets.flatten())
    expected_loss = F.cross_entropy(expected.flatten(0, 1), targets.flatten())
    actual_grad = torch.autograd.grad(actual_loss, parameters)
    expected_grad = torch.autograd.grad(expected_loss, parameters)
    double = actual.dtype == torch.float64
    rtol, atol = (1e-8, 1e-10) if double else (3e-3, 3e-4)
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    for a, b in zip(actual_grad, expected_grad, strict=True):
        torch.testing.assert_close(a, b, rtol=rtol, atol=atol)
    return {"dtype": str(actual.dtype), "device": str(actual.device),
            "max_logit_error": float((actual.detach() - expected.detach()).abs().max()),
            "max_gradient_error": max(float((a - b).abs().max()) for a, b in zip(actual_grad, expected_grad, strict=True)),
            "parameter_count": sum(p.numel() for p in parameters),
            "rtol": rtol, "atol": atol}


def main():
    torch.set_num_threads(1)
    source = Path(SOURCE["source_root"])
    for name, expected in SOURCE["files"].items():
        assert hashlib.sha256((source / name).read_bytes()).hexdigest() == expected, name
    sys.path.insert(0, str(source.parents[2]))
    from diff_gpt.domain.model.gpt import GPT

    cases = [(8, 2, 2, 5, (True, True)), (16, 2, 3, 7, (True, False, True)),
             (8, 1, 1, 1, (False,))]
    records = []
    for index, (width, heads, depth, length, rotary) in enumerate(cases):
        torch.manual_seed(31 + index)
        model = GPT(7, width, length, heads, depth, use_checkpoint=False,
                    pad_vocab_size_to=1, rope_layers=rotary).double()
        tokens, targets = torch.randint(7, (2, length)), torch.randint(7, (2, length))
        record = {"width": width, "heads": heads, "depth": depth, "length": length,
                  "rotary_layers": rotary, **compare(model, tokens, targets, model)}
        expected_count = 7 * width + depth * (4 * width**2 + 3 * width * ((8 * width + 1) // 3))
        assert record["parameter_count"] == expected_count
        records.append(record)
    torch.manual_seed(41)
    model = GPT(7, 8, 5, 2, 2, use_checkpoint=False, pad_vocab_size_to=1).cuda()
    tokens, targets = torch.randint(7, (2, 5), device="cuda"), torch.randint(7, (2, 5), device="cuda")
    compiled = torch.compile(model, mode="max-autotune", fullgraph=True, dynamic=False)
    records.append({"width": 8, "heads": 2, "depth": 2, "length": 5,
                    "rotary_layers": (True, True), "compile": "max-autotune/fullgraph/static",
                    **compare(model, tokens, targets, compiled)})
    report = {"source_commit": SOURCE["repository_commit"], "source_hashes": SOURCE["files"],
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "python": platform.python_version(), "torch": torch.__version__,
              "gpu": torch.cuda.get_device_name(), "cases": records,
              "scope": "Numerical correspondence to G4 equations, not training or a floating-point proof. Source RoPE tables are reused, including their rounding."}
    (ROOT / "sequence_model_source_check.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
