"""
gpt-mini: a minimal GPT-style transformer designed for formal verification.

Architecture choices (vs. gpt-oss in `gpt.py`):
  - Pre-LN normalization (RMSNorm before each sub-layer)
  - RMSNorm has NO trainable scale (pure normalization, ||x_norm|| = √d)
  - Multi-head causal attention (no GQA, no sliding window, no sinks)
  - QK-norm: L2-normalize Q, K + learnable per-head inverse-temperature α
  - XSA (exclusive self-attention): output ⊥ self-value v_i (arXiv:2603.09078)
  - Plain RoPE (no YaRN scaling)
  - ReLU² FFN (piecewise polynomial of degree 2, modded-nanogpt style)
  - bias=False on every linear layer
  - float32 throughout (not bfloat16)
  - Tied embedding / unembedding

Defining features kept:
  - Causal autoregressive attention
  - RoPE positional encoding
  - Pre-normalization wrapped around each sub-layer (Pre-LN)
  - Residual connections
  - Multi-head attention
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab_size:    int   = 50_257
    n_layers:      int   = 12
    n_heads:       int   = 12
    d_model:       int   = 768
    d_ff:          int   = 3_072    # 4 × d_model, standard for ReLU²
    max_seq_len:   int   = 2_048
    rope_theta:    float = 10_000.0


class RMSNorm(nn.Module):
    """Pure RMS normalization — no trainable scale.

    After `RMSNorm(x)`, ‖x_norm‖₂ = √d_model exactly, so tokens live on a
    sphere of constant radius.  This matches the canonical setup of all
    formalized clustering theorems (2312.10794, 2410.06833).
    """

    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


def rope_tables(head_dim: int, max_seq_len: int, theta: float):
    """Precompute (cos, sin) tables of shape (max_seq_len, head_dim // 2)."""
    half   = head_dim // 2
    inv_f  = theta ** (-torch.arange(half, dtype=torch.float32) / half)
    pos    = torch.arange(max_seq_len, dtype=torch.float32)
    angles = torch.outer(pos, inv_f)
    return angles.cos(), angles.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    x:   (B, H, T, head_dim)
    cos: (T, head_dim // 2)
    sin: (T, head_dim // 2)
    """
    x1, x2 = x.chunk(2, dim=-1)
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class CausalMHA(nn.Module):
    """Causal multi-head self-attention with QK-norm and XSA output.

    Forward pass:
      1. Project: q, k, v ← x · {W_q, W_k, W_v}
      2. QK-norm: q, k ← q/‖q‖, k/‖k‖              (unit L2 per head)
      3. RoPE rotation on q, k (orthogonal, preserves L2 norm)
      4. Score = ⟨q, k⟩ · e^{α_h}, masked + softmax  (bounded scores)
      5. y = attention @ v
      6. XSA: z = y - ⟨y, v̂⟩ · v̂                  (project onto v⊥ per head)
      7. Output projection: z · W_o
    """

    def __init__(self, cfg: Config):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads   = cfg.n_heads
        self.head_dim  = cfg.d_model // cfg.n_heads
        self.qkv       = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj      = nn.Linear(cfg.d_model,     cfg.d_model, bias=False)
        # Per-head learnable inverse temperature, init to ½ log(d_h)
        # so e^{α} ≈ √(d_h) at initialization (matches 1/√d_h scaling).
        self.log_alpha = nn.Parameter(
            torch.full((cfg.n_heads,), 0.5 * math.log(self.head_dim))
        )

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        # QK-norm: unit vectors per (batch, head, position).
        q = F.normalize(q, dim=-1, eps=1e-6)
        k = F.normalize(k, dim=-1, eps=1e-6)
        # RoPE is orthogonal, preserves L2 norm.
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        alpha  = self.log_alpha.exp().view(1, self.n_heads, 1, 1)
        scores = (q @ k.transpose(-2, -1)) * alpha          # ∈ [-α, α]
        mask   = torch.triu(
            torch.full((T, T), float("-inf"), device=x.device), diagonal=1
        )
        attn = F.softmax(scores + mask, dim=-1)
        y    = attn @ v                                     # (B, H, T, head_dim)
        # XSA: project y onto v^⊥ per (batch, head, position).
        v_hat = F.normalize(v, dim=-1, eps=1e-6)
        z     = y - (y * v_hat).sum(dim=-1, keepdim=True) * v_hat
        z     = z.transpose(1, 2).contiguous().view(B, T, D)
        return self.proj(z)


class ReLU2_FFN(nn.Module):
    """Two-layer feed-forward with ReLU² activation.

        FFN(x) = W_out · ReLU(W_in x)²

    Piecewise polynomial of degree 2: on {W_in x > 0} acts as a quadratic
    form in W_in x, on {W_in x ≤ 0} the corresponding coordinate is zero.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.w_in  = nn.Linear(cfg.d_model, cfg.d_ff,    bias=False)
        self.w_out = nn.Linear(cfg.d_ff,    cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_out(F.relu(self.w_in(x)).square())


class Block(nn.Module):
    """Pre-LN: norm before each sub-layer.

        x = x + Attn(Norm(x))
        x = x + FFN (Norm(x))

    With our RMSNorm-without-γ + QK-norm + bounded weights, every sub-layer
    contribution has L2 norm bounded by a constant depending only on the
    weight norms (not on x), so the residual stream grows at most linearly
    with depth.
    """
    def __init__(self, cfg: Config):
        super().__init__()
        self.norm_attn = RMSNorm()
        self.norm_ffn  = RMSNorm()
        self.attn      = CausalMHA(cfg)
        self.ffn       = ReLU2_FFN(cfg)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm_attn(x), cos, sin)
        x = x + self.ffn (self.norm_ffn (x))
        return x


class GPTMini(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg        = cfg
        self.embed      = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks     = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm_final = RMSNorm()
        self.unembed    = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        # Tied weights.
        self.unembed.weight = self.embed.weight
        # Precomputed RoPE tables.
        head_dim = cfg.d_model // cfg.n_heads
        cos, sin = rope_tables(head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: (B, T) int64.  Returns logits: (B, T, vocab_size)."""
        B, T = tokens.shape
        assert T <= self.cfg.max_seq_len
        x   = self.embed(tokens)
        cos = self.cos[:T]
        sin = self.sin[:T]
        for block in self.blocks:
            x = block(x, cos, sin)
        return self.unembed(self.norm_final(x))


if __name__ == "__main__":
    cfg   = Config()
    model = GPTMini(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"gpt-mini: {n_params / 1e6:.1f}M parameters")
    print(f"  layers: {cfg.n_layers}")
    x = torch.randint(0, cfg.vocab_size, (2, 128))
    y = model(x)
    print(f"input: {tuple(x.shape)}  output: {tuple(y.shape)}")
