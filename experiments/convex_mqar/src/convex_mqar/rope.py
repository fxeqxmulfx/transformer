"""A two-layer pre-norm causal Transformer with learned embeddings and RoPE."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class RotaryAttention(nn.Module):
    def __init__(self, width, heads, base):
        super().__init__()
        self.heads = heads
        self.head_width = width // heads
        if width % heads or self.head_width % 2:
            raise ValueError("RoPE needs an even head width")
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.output = nn.Linear(width, width, bias=False)
        self.register_buffer("inverse_frequency",
                             base ** (-torch.arange(0, self.head_width, 2).float()
                                      / self.head_width), persistent=False)

    def rotate(self, x):
        positions = torch.arange(x.shape[-2], device=x.device, dtype=torch.float32)
        angles = positions[:, None] * self.inverse_frequency[None, :]
        cosine, sine = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
        even, odd = x[..., 0::2], x[..., 1::2]
        return torch.stack((even * cosine - odd * sine,
                            even * sine + odd * cosine), dim=-1).flatten(-2)

    def forward(self, x):
        batch, length, width = x.shape
        qkv = self.qkv(x).reshape(batch, length, 3, self.heads, self.head_width)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attended = F.scaled_dot_product_attention(self.rotate(q), self.rotate(k), v,
                                                 dropout_p=0.0, is_causal=True)
        return self.output(attended.transpose(1, 2).reshape(batch, length, width))


class Block(nn.Module):
    def __init__(self, width, heads, ratio, base):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.attention = RotaryAttention(width, heads, base)
        self.mlp = nn.Sequential(nn.Linear(width, ratio * width),
                                 nn.GELU(approximate="tanh"),
                                 nn.Linear(ratio * width, width))

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class RopeTransformer(nn.Module):
    def __init__(self, vocab, width, layers=2, heads=1, mlp_ratio=4, rope_base=10_000.0):
        super().__init__()
        self.embedding = nn.Embedding(vocab, width)
        self.blocks = nn.ModuleList(Block(width, heads, mlp_ratio, rope_base)
                                    for _ in range(layers))
        self.final_norm = nn.LayerNorm(width)
        self.apply(self.initialize)
        for block in self.blocks:
            nn.init.normal_(block.attention.output.weight, std=0.02 / math.sqrt(2 * layers))
            nn.init.normal_(block.mlp[-1].weight, std=0.02 / math.sqrt(2 * layers))

    @staticmethod
    def initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, tokens, positions=None):
        features = self.embedding(tokens)
        for block in self.blocks:
            features = block(features)
        features = self.final_norm(features)
        if positions is not None:
            # This only avoids computing unused output logits. It does not
            # change attention, causality, or access to the full raw sequence.
            features = features.gather(1, positions[..., None].expand(-1, -1, features.shape[-1]))
        return F.linear(features, self.embedding.weight)
