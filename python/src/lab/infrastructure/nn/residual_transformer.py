"""No-cache port of diff_gpt ResidualGPT; see domain.residual_transformer.

Common parameter names, initializer draw order and both depth recurrences
match the source. Historical SourceTransformer remains unchanged. Shared
constant prefixes are valid only while every earlier XSA head is above its
clipping epsilon; the mask can lose validity but cannot regain it.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .centered_xsa import centered_projection, XSA_EPS
from .depth_mixing import DepthMixer
from .source_transformer import SourceAttention, SourceFeedForward, SourceTransformer, rotate


class CenteredAttention(SourceAttention):
    def forward(self, x, cos, sin):
        batch, length, width = x.shape
        q, k, v = self.c_attn(x).chunk(3, dim=-1)
        q, k, v = (z.view(batch, length, self.n_head, self.head_dim).transpose(1, 2)
                   for z in (q, k, v))
        q, k = rotate(self.q_norm(q), cos, sin), rotate(self.k_norm(k), cos, sin)
        delta = v - v[:, :, :1]
        average = F.scaled_dot_product_attention(q, k, delta, is_causal=True)
        y = centered_projection(average, delta, v)
        y = torch.cat((torch.zeros_like(y[:, :, :1]), y[:, :, 1:]), dim=2)
        valid = (torch.linalg.vector_norm(v[:, :, 0], dim=-1) > XSA_EPS).all(dim=1)
        return self.c_proj(y.transpose(1, 2).contiguous().view(batch, length, width)), valid


class ResidualBlock(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.ln_1 = nn.RMSNorm(spec.width, elementwise_affine=False)
        self.sa = CenteredAttention(spec)
        self.ln_2 = nn.RMSNorm(spec.width, elementwise_affine=False)
        self.ffwd = SourceFeedForward(spec)
        self.attn_out_norm = nn.RMSNorm(spec.width, elementwise_affine=False)
        self.ffn_out_norm = nn.RMSNorm(spec.width, elementwise_affine=False)

    def attention(self, x, cos, sin):
        value, valid = self.sa(self.ln_1(x), cos, sin)
        return self.attn_out_norm(value), valid

    def feed_forward(self, x):
        return self.ffn_out_norm(self.ffwd(self.ln_2(x)))


class ResidualTransformer(nn.Module):
    def __init__(self, spec, vocab):
        super().__init__()
        self.spec, self.vocab = spec, vocab
        self.token_embedding_table = nn.Embedding(vocab, spec.width)
        dim = spec.width // spec.heads
        exponents = torch.arange(0, dim, 2, dtype=torch.float32) / dim
        angles = torch.outer(torch.arange(spec.context, dtype=torch.float32), spec.rope_base**-exponents)
        self.register_buffer("rope_cos", angles.cos(), persistent=False)
        self.register_buffer("rope_sin", angles.sin(), persistent=False)
        self.blocks = nn.ModuleList(ResidualBlock(spec) for _ in range(spec.depth))
        self.ln_f = nn.RMSNorm(spec.width, elementwise_affine=False)
        self.embedding_norm = nn.RMSNorm(spec.width, elementwise_affine=False)
        self.lm_head = nn.Linear(spec.width, vocab, bias=False)
        self.lm_head.weight = self.token_embedding_table.weight
        self.apply(SourceTransformer._init_weights)
        for name, parameter in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(parameter, std=.02 / math.sqrt(2 * spec.depth))
        self.branch_scale = 1 / math.sqrt(2 * spec.depth) if spec.mixing == "scaled" else 1.
        self.residual_mixers = nn.ModuleList(
            [] if spec.mixing in ("add", "scaled") else
            [DepthMixer(spec.width, spec.mixing) for _ in range(2 * spec.depth)])

    def aggregate(self, index, sources):
        return sources[0] if index == 0 else self.residual_mixers[index - 1](torch.stack(sources, dim=-2))

    def forward(self, inputs):
        length = inputs.shape[1]
        if length > self.spec.context:
            raise ValueError("Input exceeds the configured source RoPE context")
        cos, sin = self.rope_cos[:length], self.rope_sin[:length]
        x = self.embedding_norm(self.token_embedding_table(inputs))
        constant = (inputs != inputs[:, :1]).cumsum(dim=1) == 0
        sources = [x]
        for index, block in enumerate(self.blocks):
            if self.residual_mixers:
                x = self.aggregate(2 * index, sources)
            branch, valid = block.attention(x, cos, sin)
            constant = constant & valid[:, None]
            if self.spec.mixing == "add":
                # Match GPT's post-block prefix sharing and its gradient graph.
                x = x + branch
                x = x + block.feed_forward(x)
                x = torch.where(constant[..., None], x[:, :1], x)
                continue
            branch = torch.where(constant[..., None], branch[:, :1], branch)
            if self.residual_mixers:
                sources.append(branch)
                x = self.aggregate(2 * index + 1, sources)
            else:
                x = x + self.branch_scale * branch
            branch = block.feed_forward(x)
            branch = torch.where(constant[..., None], branch[:, :1], branch)
            if self.residual_mixers:
                sources.append(branch)
            else:
                x = x + self.branch_scale * branch
        if self.residual_mixers:
            x = self.aggregate(2 * self.spec.depth, sources)
        if self.spec.mixing != "add":
            x = torch.where(constant[..., None], x[:, :1], x)
        return self.lm_head(self.ln_f(x))
