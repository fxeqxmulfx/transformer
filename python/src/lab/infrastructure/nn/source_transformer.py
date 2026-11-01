"""Port of diff_gpt/domain/model at b98c665b; see domain.source_transformer.

Module names and initializer draw order match the source. The explicit
first-position zero uses concatenation instead of an in-place slice; no
cache, activation checkpointing or text-generation interface is included.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


def rotate(x, cos, sin):
    first, second = x.chunk(2, dim=-1)
    return torch.cat((first * cos - second * sin, first * sin + second * cos), dim=-1)


class SourceAttention(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.n_head, self.head_dim = spec.heads, spec.width // spec.heads
        self.use_rope = True
        self.c_attn = nn.Linear(spec.width, 3 * spec.width, bias=False)
        self.c_proj = nn.Linear(spec.width, spec.width, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, elementwise_affine=False)
        self.k_norm = nn.RMSNorm(self.head_dim, elementwise_affine=False)

    def forward(self, x, cos, sin):
        batch, length, width = x.shape
        q, k, v = self.c_attn(x).chunk(3, dim=-1)
        q, k, v = (z.view(batch, length, self.n_head, self.head_dim).transpose(1, 2)
                   for z in (q, k, v))
        q, k = rotate(self.q_norm(q), cos, sin), rotate(self.k_norm(k), cos, sin)
        own = F.normalize(v, dim=-1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y - (y * own).sum(dim=-1, keepdim=True) * own
        y = torch.cat((torch.zeros_like(y[:, :, :1]), y[:, :, 1:]), dim=2)
        return self.c_proj(y.transpose(1, 2).contiguous().view(batch, length, width))


class SourceFeedForward(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.c_fc = nn.Linear(spec.width, 2 * spec.hidden, bias=False)
        self.c_proj = nn.Linear(spec.hidden, spec.width, bias=False)

    def forward(self, x):
        gate, value = self.c_fc(x).chunk(2, dim=-1)
        return self.c_proj(F.silu(gate) * value)


class SourceBlock(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.ln_1 = nn.RMSNorm(spec.width, elementwise_affine=False)
        self.sa = SourceAttention(spec)
        self.ln_2 = nn.RMSNorm(spec.width, elementwise_affine=False)
        self.ffwd = SourceFeedForward(spec)
        self.attn_out_norm = nn.RMSNorm(spec.width, elementwise_affine=False)
        self.ffn_out_norm = nn.RMSNorm(spec.width, elementwise_affine=False)

    def forward(self, x, cos, sin):
        x = x + self.attn_out_norm(self.sa(self.ln_1(x), cos, sin))
        return x + self.ffn_out_norm(self.ffwd(self.ln_2(x)))


class SourceTransformer(nn.Module):
    def __init__(self, spec, vocab):
        super().__init__()
        self.spec, self.vocab = spec, vocab
        self.token_embedding_table = nn.Embedding(vocab, spec.width)
        dim = spec.width // spec.heads
        exponents = torch.arange(0, dim, 2, dtype=torch.float32) / dim
        angles = torch.outer(torch.arange(spec.context, dtype=torch.float32), spec.rope_base**-exponents)
        self.register_buffer("rope_cos", angles.cos(), persistent=False)
        self.register_buffer("rope_sin", angles.sin(), persistent=False)
        self.blocks = nn.ModuleList(SourceBlock(spec) for _ in range(spec.depth))
        self.ln_f = nn.RMSNorm(spec.width, elementwise_affine=False)
        self.embedding_norm = nn.RMSNorm(spec.width, elementwise_affine=False)
        self.lm_head = nn.Linear(spec.width, vocab, bias=False)
        self.lm_head.weight = self.token_embedding_table.weight
        self.apply(self._init_weights)
        for name, parameter in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(parameter, std=.02 / math.sqrt(2 * spec.depth))

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=.02)

    def forward(self, inputs):
        length = inputs.shape[1]
        if length > self.spec.context:
            raise ValueError("Input exceeds the configured source RoPE context")
        x = self.embedding_norm(self.token_embedding_table(inputs))
        for block in self.blocks:
            x = block(x, self.rope_cos[:length], self.rope_sin[:length])
        return self.lm_head(self.ln_f(x))
