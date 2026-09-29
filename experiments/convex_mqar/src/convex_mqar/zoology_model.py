"""Portable Figure 2 attention model from Zoology's pinned iclr24 tag.

Preserves the two residual/LN blocks, Identity state mixer, biases, dropout,
initialization, and tied readout. Causal SDPA replaces explicit softmax.
RoPE is an explicit extension; positions='learned' uses the source positions.
"""

from functools import partial
import math

import torch
from torch import nn
from torch.nn import functional as F

from .rope import RotaryAttention


def initialize(module, n_layers):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        nn.init.normal_(module.weight, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)
    for name, parameter in module.named_parameters():
        if "out_proj.weight" in name or "fc2.weight" in name:
            nn.init.normal_(parameter, std=0.02/math.sqrt(2*n_layers))


class TokenEmbeddings(nn.Module):
    def __init__(self, vocab, width, length, positions):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab, width)
        self.max_position_embeddings = length if positions == "learned" else 0
        if self.max_position_embeddings:
            self.position_embeddings = nn.Embedding(length, width)

    def forward(self, tokens):
        features = self.word_embeddings(tokens)
        if self.max_position_embeddings:
            features = features+self.position_embeddings(torch.arange(tokens.shape[1], device=tokens.device))
        return features


class Attention(nn.Module):
    rotate = RotaryAttention.rotate

    def __init__(self, width, heads, base, positions, dropout):
        super().__init__()
        if width % heads or (width//heads) % 2:
            raise ValueError("Use an even head width")
        self.heads, self.head_width = heads, width//heads
        self.positions, self.dropout = positions, dropout
        self.Wqkv = nn.Linear(width, 3*width)
        self.out_proj = nn.Linear(width, width)
        self.register_buffer("inverse_frequency", base**(
            -torch.arange(0, self.head_width, 2).float()/self.head_width), persistent=False)

    def forward(self, x):
        batch, length, width = x.shape
        qkv = self.Wqkv(x).reshape(batch, length, 3, self.heads, self.head_width)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        if self.positions == "rope":
            q, k = self.rotate(q), self.rotate(k)
        output = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                                dropout_p=self.dropout if self.training else 0.)
        return self.out_proj(output.transpose(1, 2).reshape(batch, length, width))


class Block(nn.Module):
    def __init__(self, width, heads, base, positions, dropout, index):
        super().__init__()
        self.sequence_mixer = Attention(width, heads, base, positions, dropout)
        self.state_mixer = nn.Identity()
        self.dropout1 = nn.Dropout(dropout if index == 0 else 0.)
        self.norm1 = nn.LayerNorm(width)
        self.dropout2 = nn.Dropout(0.)
        self.norm2 = nn.LayerNorm(width)

    def forward(self, x, residual):
        dropped = self.dropout1(x)
        residual = dropped if residual is None else residual+dropped
        x = self.sequence_mixer(self.norm1(residual))
        residual = residual+self.dropout2(x)
        return self.state_mixer(self.norm2(residual)), residual


class Backbone(nn.Module):
    def __init__(self, vocab, width, length, layers, heads, base, positions, dropout):
        super().__init__()
        self.embeddings = TokenEmbeddings(vocab, width, length, positions)
        self.layers = nn.ModuleList(Block(width, heads, base, positions, dropout, r)
                                    for r in range(layers))
        self.ln_f = nn.LayerNorm(width)
        self.apply(partial(initialize, n_layers=layers))

    def forward(self, tokens):
        x, residual = self.embeddings(tokens), None
        for layer in self.layers:
            x, residual = layer(x, residual)
        return self.ln_f(residual+x)


class ZoologyTransformer(nn.Module):
    def __init__(self, vocab, width, length, layers=2, heads=1, rope_base=10_000.,
                 positions="rope", dropout=0.1):
        super().__init__()
        if positions not in ("rope", "learned"):
            raise ValueError("Choose rope or learned positions")
        self.backbone = Backbone(vocab, width, length, layers, heads, rope_base, positions, dropout)
        self.lm_head = nn.Linear(width, vocab, bias=False)
        self.apply(partial(initialize, n_layers=layers))
        self.lm_head.weight = self.backbone.embeddings.word_embeddings.weight

    def forward(self, tokens, positions=None):
        features = self.backbone(tokens)
        if positions is not None:
            features = features.gather(1, positions[..., None].expand(-1, -1, features.shape[-1]))
        return self.lm_head(features)
