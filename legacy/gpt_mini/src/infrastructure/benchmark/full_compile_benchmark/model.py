"""Functional GPTMini with optional zero probes at all Linear outputs.

Differentiating zero additive probes yields exactly the output derivatives
collected by AdaFisher's backward hooks. Returning activation second moments
as auxiliary values avoids side effects inside the differentiation transform.
"""

import torch
from torch.nn import functional as F

from gpt_mini.gpt_mini import apply_rope


class FunctionalSparsemax(torch.autograd.Function):
    generate_vmap_rule = True

    @staticmethod
    def forward(scores):
        # No nested legacy autograd.Function in the torch.func transform.
        maximum = scores.amax(-1, keepdim=True)
        finite = maximum.isfinite()
        centered = scores - torch.where(finite, maximum, torch.zeros_like(maximum))
        ordered = centered.sort(-1, descending=True).values
        cumulative = ordered.cumsum(-1)
        ranks = torch.arange(1, scores.size(-1) + 1, device=scores.device, dtype=scores.dtype)
        size = (1 + ranks * ordered > cumulative).sum(-1, keepdim=True).clamp_min(1)
        threshold = (cumulative.gather(-1, size - 1) - 1) / size
        output = (centered - threshold).clamp_min(0)
        return torch.where(finite, output, torch.full_like(output, float("nan")))

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(output > 0)

    @staticmethod
    def backward(ctx, gradient):
        (active,) = ctx.saved_tensors
        count = active.sum(-1, keepdim=True).clamp_min(1)
        mean = (gradient * active).sum(-1, keepdim=True) / count
        return active * (gradient - mean)


def forward(config, attention, parameters, cos, sin, tokens, probes=()):
    batch, length = tokens.shape
    heads, width = config.n_heads, config.d_model
    head_dim = width // heads
    h_factors = []

    def linear(inputs, weight):
        output = F.linear(inputs, weight)
        if probes:
            index = len(h_factors)
            h_factors.append(inputs.detach().flatten(0, 1).square().sum(0))
            output = output + probes[index]
        return output

    x = F.embedding(tokens, parameters["embed.weight"])
    cos, sin = cos[:length], sin[:length]
    for layer in range(config.n_layers):
        prefix = f"blocks.{layer}."
        normalized = F.rms_norm(x, (width,), eps=1e-5)
        q, k, v = linear(normalized, parameters[prefix + "attn.qkv.weight"]).chunk(3, -1)
        q, k, v = (t.view(batch, length, heads, head_dim).transpose(1, 2) for t in (q, k, v))
        q = apply_rope(F.normalize(q, dim=-1, eps=1e-6), cos, sin)
        k = apply_rope(F.normalize(k, dim=-1, eps=1e-6), cos, sin)
        alpha = parameters[prefix + "attn.log_alpha"].exp().view(1, heads, 1, 1)
        scores = (q @ k.transpose(-2, -1)) * alpha
        if attention == "softmax":
            mask = torch.full((length, length), float("-inf"), device=x.device).triu(1)
            probabilities = F.softmax(scores + mask, -1)
        else:
            mask = torch.ones((length, length), device=x.device, dtype=torch.bool).triu(1)
            probabilities = FunctionalSparsemax.apply(scores.masked_fill(mask, float("-inf")))
        y = probabilities @ v
        v_hat = F.normalize(v, dim=-1, eps=1e-6)
        z = y - (y * v_hat).sum(-1, keepdim=True) * v_hat
        z = z.transpose(1, 2).contiguous().view(batch, length, width)
        x = x + linear(z, parameters[prefix + "attn.proj.weight"])
        normalized = F.rms_norm(x, (width,), eps=1e-5)
        hidden = F.relu(linear(normalized, parameters[prefix + "ffn.w_in.weight"])).square()
        x = x + linear(hidden, parameters[prefix + "ffn.w_out.weight"])
    logits = linear(F.rms_norm(x, (width,), eps=1e-5), parameters["embed.weight"])
    return logits, tuple(h_factors)


def output_shapes(config):
    return [dimension for _ in range(config.n_layers)
            for dimension in (3 * config.d_model, config.d_model, config.d_ff, config.d_model)] + [config.vocab_size]


def linear_names(config):
    return [f"blocks.{i}.{suffix}.weight" for i in range(config.n_layers)
            for suffix in ("attn.qkv", "attn.proj", "ffn.w_in", "ffn.w_out")] + ["embed.weight"]
