"""Model blocks: a decoder-only transformer composed of swappable parts.

Structural choices have no defaults, so an experiment file states its whole
architecture; only numerical constants (epsilons, RoPE base) default to their
standard values. Three historical models are compositions of these blocks:

- GPTMini (`experiments/archive/gpt_mini/gpt_mini.py`): PreNorm,
  parameter-free RMSNorm, FusedQKV, QKNorm, Softmax, XSA, ReLU2, RoPE, Tied,
  final RMSNorm, Normal(0.02);
- the openai/grok reference: PostNorm, LayerNorm, PerHeadQKV, ScaledDot,
  Softmax, ReLU, Sinusoidal, Untied, no final norm, TorchDefault;
- the RoPE transformer of the convex MQAR comparison
  (`experiments/convex_mqar/src/convex_mqar/rope.py`): PreNorm, LayerNorm,
  FusedQKV, ScaledDot, fused Softmax, GELU(tanh) with biases, interleaved
  RoPE, Tied, final LayerNorm, ScaledResidual(0.02).
"""

from dataclasses import dataclass

from .spec import Spec, require, require_kind


@dataclass(frozen=True)
class Norm(Spec, kind=True):
    """Normalization of a residual-stream vector."""


@dataclass(frozen=True)
class RMSNorm(Norm):
    """x / rms(x); `scale` adds a learned per-channel gain."""
    eps: float = 1e-5
    scale: bool = False

    def check(self):
        require(self.eps > 0, "RMSNorm epsilon must be positive")


@dataclass(frozen=True)
class LayerNorm(Norm):
    """(x - mean) / std; `affine` adds a learned gain and bias."""
    eps: float = 1e-5
    affine: bool = True

    def check(self):
        require(self.eps > 0, "LayerNorm epsilon must be positive")


@dataclass(frozen=True)
class Positions(Spec, kind=True):
    """How token order enters the model."""


@dataclass(frozen=True)
class RoPE(Positions):
    """Rotate queries and keys by position inside every attention layer.

    A head's channel i is paired with channel i + head/2 (the two halves), or
    with `interleaved` channel 2i with channel 2i + 1.
    """
    theta: float = 10_000.0
    interleaved: bool = False

    def check(self):
        require(self.theta > 1, "RoPE base must exceed one")


@dataclass(frozen=True)
class Sinusoidal(Positions):
    """Add the fixed sine/cosine table of Vaswani et al. to token embeddings."""
    base: float = 10_000.0

    def check(self):
        require(self.base > 1, "Sinusoid base must exceed one")


@dataclass(frozen=True)
class NoPositions(Positions):
    """Causal masking is the only source of order."""


@dataclass(frozen=True)
class Projections(Spec, kind=True):
    """How queries, keys and values are computed from the input."""


@dataclass(frozen=True)
class FusedQKV(Projections):
    """One width-by-3-width matrix whose output is split into heads."""
    bias: bool = False


@dataclass(frozen=True)
class PerHeadQKV(Projections):
    """Separate width-by-head matrices per head and role; heads run one by one."""
    bias: bool = False


@dataclass(frozen=True)
class Scores(Spec, kind=True):
    """The attention logit of a query-key pair."""


@dataclass(frozen=True)
class ScaledDot(Scores):
    """q . k / sqrt(head width)."""


@dataclass(frozen=True)
class QKNorm(Scores):
    """Unit q . unit k times a learned per-head e^alpha, alpha = log(head width) / 2 at start."""
    eps: float = 1e-6

    def check(self):
        require(self.eps > 0, "QKNorm epsilon must be positive")


@dataclass(frozen=True)
class Weights(Spec, kind=True):
    """Causal normalization of a row of scores into attention weights."""


@dataclass(frozen=True)
class Softmax(Weights):
    """exp(s) / sum exp(s) over the visible prefix.

    `fused` computes the attention it weights by PyTorch's fused scaled
    dot-product attention: the same function, rounded otherwise. It needs
    ScaledDot scores, whose scale the kernel applies itself.
    """
    fused: bool = False


@dataclass(frozen=True)
class Sparsemax(Weights):
    """Euclidean projection of the visible scores onto the simplex; exact zeros."""


@dataclass(frozen=True)
class XSA(Spec):
    """Exclusive self-attention: remove the output component along the own value."""
    eps: float = 1e-6

    def check(self):
        require(self.eps > 0, "XSA epsilon must be positive")


@dataclass(frozen=True)
class Attention(Spec):
    """Causal multi-head self-attention."""
    heads: int
    projections: Projections
    scores: Scores
    weights: Weights
    exclusive: XSA | None
    output_bias: bool = False

    def check(self):
        require(self.heads >= 1, "Attention needs at least one head")
        require_kind(self.projections, Projections, "projections")
        require_kind(self.scores, Scores, "scores")
        require_kind(self.weights, Weights, "weights")
        if self.exclusive is not None:
            require_kind(self.exclusive, XSA, "exclusive")
        if isinstance(self.weights, Softmax) and self.weights.fused:
            require(isinstance(self.scores, ScaledDot), "Fused softmax attention needs ScaledDot scores")


@dataclass(frozen=True)
class Activation(Spec, kind=True):
    """The pointwise nonlinearity of a feed-forward layer."""


@dataclass(frozen=True)
class ReLU(Activation):
    """max(x, 0)."""


@dataclass(frozen=True)
class ReLU2(Activation):
    """max(x, 0) squared."""


@dataclass(frozen=True)
class GELU(Activation):
    """x Phi(x), the exact erf form; `tanh` its tanh approximation."""
    tanh: bool = False


@dataclass(frozen=True)
class FFN(Spec):
    """Two matrices around an activation; the hidden width is multiplier * width."""
    activation: Activation
    multiplier: int = 4
    bias: bool = False

    def check(self):
        require_kind(self.activation, Activation, "activation")
        require(self.multiplier >= 1, "FFN multiplier must be positive")


@dataclass(frozen=True)
class Residual(Spec, kind=True):
    """Where a block normalizes relative to its residual connection."""


@dataclass(frozen=True)
class PreNorm(Residual):
    """x + f(norm(x))."""


@dataclass(frozen=True)
class PostNorm(Residual):
    """norm(x + f(x))."""


@dataclass(frozen=True)
class Block(Spec):
    """Attention then feed-forward, each wrapped by the residual rule."""
    attention: Attention
    ffn: FFN
    norm: Norm
    residual: Residual

    def check(self):
        require_kind(self.attention, Attention, "attention")
        require_kind(self.ffn, FFN, "ffn")
        require_kind(self.norm, Norm, "norm")
        require_kind(self.residual, Residual, "residual")


@dataclass(frozen=True)
class Readout(Spec, kind=True):
    """How the final residual stream becomes vocabulary logits."""


@dataclass(frozen=True)
class Tied(Readout):
    """Logits are products with the embedding matrix."""


@dataclass(frozen=True)
class Untied(Readout):
    """A separate output matrix."""
    bias: bool = False


@dataclass(frozen=True)
class Init(Spec, kind=True):
    """Parameter initialization after the model seed is set."""


@dataclass(frozen=True)
class TorchDefault(Init):
    """Each module's PyTorch default, drawn in construction order."""


@dataclass(frozen=True)
class Normal(Init):
    """After the defaults, redraw every matrix (ndim >= 2) from N(0, std^2) in parameter order."""
    std: float = 0.02

    def check(self):
        require(self.std > 0, "Initialization scale must be positive")


@dataclass(frozen=True)
class ScaledResidual(Init):
    """Normal matrices, zero biases, and the writes into the residual stream scaled by depth.

    After the defaults, every matrix (ndim >= 2) is redrawn from N(0, std^2)
    in parameter order and every bias of a linear map is set to zero; then,
    block by block, the two matrices that write into the residual stream (the
    attention output, then the feed-forward output) are redrawn at
    std / sqrt(2 depth). A tied readout draws no matrix of its own. The
    scaling is GPT-2's, by 1 / sqrt(N) for N residual layers (Radford et al.
    2019, Section 2.3; not in `papers/`).
    """
    std: float = 0.02

    def check(self):
        require(self.std > 0, "Initialization scale must be positive")


@dataclass(frozen=True)
class Transformer(Spec):
    """Embedding, `depth` copies of `block`, optional final norm, readout."""
    width: int
    depth: int
    block: Block
    positions: Positions
    readout: Readout
    final_norm: Norm | None
    init: Init
    context: int

    def check(self):
        require(self.width >= 1 and self.depth >= 1 and self.context >= 1,
                "Width, depth and context must be positive")
        require_kind(self.block, Block, "block")
        require_kind(self.positions, Positions, "positions")
        require_kind(self.readout, Readout, "readout")
        require_kind(self.init, Init, "init")
        if self.final_norm is not None:
            require_kind(self.final_norm, Norm, "final_norm")
        heads = self.block.attention.heads
        require(self.width % heads == 0, f"Width {self.width} is not divisible by {heads} heads")
        if isinstance(self.positions, RoPE):
            require(self.width // heads % 2 == 0, "RoPE needs an even head width")

    @property
    def head_width(self):
        return self.width // self.block.attention.heads
