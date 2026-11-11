//! The model's sizes and where each parameter tensor lies in the flat parameter vector, which
//! the reference model, the fast model, the optimizer and the checkpoints share.
//!
//! A linear map is stored as `in` rows of `out` floats, so that `y = x W` is a product of
//! row-major matrices; PyTorch stores the transpose. The entries are independent and
//! identically distributed at initialization, so the law of the model is the same.

/// The sizes of a decoder of openai/grok `grok/transformer.py`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Dims {
    /// Tokens.
    pub vocab: usize,
    /// Width.
    pub d: usize,
    /// Attention heads, which divide the width.
    pub heads: usize,
    /// Decoder blocks.
    pub layers: usize,
    /// Hidden width of a block's feed-forward network: four times the width in the source.
    pub dff: usize,
    /// Tokens read.
    pub ctx: usize,
}

/// Where the tensors of one decoder block start.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Block {
    /// Queries: `d` rows of `d`, head `h` in columns `h dk..(h + 1) dk`.
    pub wq: usize,
    /// Keys, then values: `d` rows of `2 d`.
    pub wkv: usize,
    /// The map after the heads are concatenated: `d` rows of `d`.
    pub wo: usize,
    /// Gain of the normalization after attention.
    pub g1: usize,
    /// Its bias.
    pub b1: usize,
    /// First feed-forward map: `d` rows of `dff`.
    pub w1: usize,
    /// Second: `dff` rows of `d`.
    pub w2: usize,
    /// Gain of the normalization after the feed-forward network.
    pub g2: usize,
    /// Its bias.
    pub b2: usize,
}

impl Dims {
    /// The model of arXiv:2201.02177: 2 layers, width 128, 4 heads, the author's 239 tokens.
    pub const PAPER: Dims = Dims { vocab: 239, d: 128, heads: 4, layers: 2, dff: 512, ctx: 6 };

    /// Width of a head.
    pub fn dk(&self) -> usize {
        self.d / self.heads
    }

    /// Parameters of one block.
    pub fn block_len(&self) -> usize {
        let d = self.d;
        d * d + d * 2 * d + d * d + 2 * d + d * self.dff + self.dff * d + 2 * d
    }

    /// Parameters of the model: the embedding, the blocks, the readout.
    pub fn len(&self) -> usize {
        self.vocab * self.d + self.layers * self.block_len() + self.d * self.vocab
    }

    /// Where the embedding starts: `vocab` rows of `d`.
    pub fn emb(&self) -> usize {
        0
    }

    /// Where the tensors of block `l` start.
    pub fn block(&self, l: usize) -> Block {
        let d = self.d;
        let wq = self.vocab * d + l * self.block_len();
        let wkv = wq + d * d;
        let wo = wkv + d * 2 * d;
        let g1 = wo + d * d;
        let b1 = g1 + d;
        let w1 = b1 + d;
        let w2 = w1 + d * self.dff;
        let g2 = w2 + self.dff * d;
        let b2 = g2 + d;
        Block { wq, wkv, wo, g1, b1, w1, w2, g2, b2 }
    }

    /// Where the readout starts: `d` rows of `vocab`.
    pub fn wout(&self) -> usize {
        self.vocab * self.d + self.layers * self.block_len()
    }

    /// The positional encoding of `Transformer._position_encoding`, `ctx` rows of `d`: computed
    /// in doubles and rounded to `f32`, as the model's buffer is after `.float()`.
    pub fn positions(&self) -> Vec<f32> {
        let d = self.d;
        let mut pe = vec![0f32; self.ctx * d];
        for pos in 0..self.ctx {
            for i in 0..d {
                let value = if i % 2 == 0 {
                    (pos as f64 / 10000f64.powf(i as f64 / d as f64)).sin()
                } else {
                    (pos as f64 / 10000f64.powf((i - 1) as f64 / d as f64)).cos()
                };
                pe[pos * d + i] = value as f32;
            }
        }
        pe
    }
}

/// The variance floor of `torch.nn.LayerNorm`, added under the square root.
pub const LN_EPS: f64 = 1e-5;
