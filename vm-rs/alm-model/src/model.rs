//! The transformer itself, on burn.
//!
//! Ported from `VanillaTransformer` in `transformer_vm/model/transformer.py`
//! and its C++ twin.  The shape of the computation is unusual for a tensor
//! framework and worth stating: generation is one token at a time with no
//! batch, so every operation here is a matrix-vector product against a residual
//! stream 36 wide.  burn's part is that linear algebra; the attention is not
//! a tensor operation at all but a query into a convex hull, which is why the
//! forward pass hands the projections out to `alm_hull` and takes a vector back.
//!
//! The element type is `f64` throughout, as it is in the original
//! (`torch.set_default_dtype(torch.float64)`).  It is not an implementation
//! detail: the exactness the construction claims is exactness of integers below
//! `2^53`, and nothing about it survives a narrower float.

use burn::module::Param;
use burn::nn::{Linear, LinearConfig};
use burn::prelude::Backend;
use burn::tensor::{Tensor, TensorData};

use crate::cache::KvCache;
use crate::weights::{RawModel, Shapes};

/// The four projections of one layer.
pub struct LayerWeights<B: Backend> {
    qkv: Linear<B>,
    out: Linear<B>,
    ff_in: Linear<B>,
    ff_out: Linear<B>,
}

/// The compiled transformer.
pub struct Alm<B: Backend> {
    pub shapes: Shapes,
    pub tokens: Vec<String>,
    embedding: Vec<f64>,
    layers: Vec<LayerWeights<B>>,
    head: Linear<B>,
    device: B::Device,
}

/// Build a `Linear` from a row-major `[rows, cols]` matrix, without a bias.
///
/// burn stores a linear layer's weight as `[d_input, d_output]` and computes
/// `O = I W`, while the file stores `[d_output, d_input]` and computes `y = W x`,
/// so the matrix is transposed once here rather than at every token.
fn linear_from<B: Backend>(w: &[f64], rows: usize, cols: usize, device: &B::Device) -> Linear<B> {
    let data = TensorData::new(w.to_vec(), [rows, cols]);
    let weight = Tensor::<B, 2>::from_data(data, device).transpose();
    let mut layer = LinearConfig::new(cols, rows).with_bias(false).init(device);
    layer.weight = Param::from_tensor(weight);
    layer
}

impl<B: Backend> Alm<B> {
    pub fn from_raw(raw: &RawModel, device: &B::Device) -> Alm<B> {
        let s = raw.shapes;
        let (d, f, v) = (s.d_model, s.d_ffn, s.vocab);
        let layers = raw
            .layers
            .iter()
            .map(|l| LayerWeights {
                qkv: linear_from(&l.qkv, 3 * d, d, device),
                out: linear_from(&l.out, d, d, device),
                ff_in: linear_from(&l.ff_in, 2 * f, d, device),
                ff_out: linear_from(&l.ff_out, d, f, device),
            })
            .collect();
        Alm {
            shapes: s,
            tokens: raw.tokens.clone(),
            embedding: raw.embedding.clone(),
            layers,
            head: linear_from(&raw.head, v, d, device),
            device: device.clone(),
        }
    }

    fn row(&self, x: &[f64]) -> Tensor<B, 2> {
        Tensor::from_data(TensorData::new(x.to_vec(), [1, x.len()]), &self.device)
    }

    fn plain(t: Tensor<B, 2>) -> Vec<f64> {
        t.into_data().to_vec::<f64>().expect("the backend is f64")
    }

    /// The residual stream at the start of a position: the token embedding plus
    /// the three position features the compiler assumes are there.
    ///
    /// `1/log 2 - 1/log(pos + 2)` is the recency feature the latest-write
    /// perturbation is built from, and `pos * pos` is what makes a parabolic
    /// key of the position.
    pub fn embed(&self, token: usize, pos: usize) -> Vec<f64> {
        let d = self.shapes.d_model;
        let mut x = self.embedding[token * d..(token + 1) * d].to_vec();
        let p = pos as f64;
        x[0] += p;
        x[1] += 1.0 / std::f64::consts::LN_2 - 1.0 / ((p + 2.0).ln());
        x[2] += p * p;
        x
    }

    /// One position through the whole stack, updating the cache as it goes.
    pub fn forward(&self, token: usize, pos: usize, cache: &mut KvCache) -> Vec<f64> {
        let (d, f) = (self.shapes.d_model, self.shapes.d_ffn);
        let mut x = self.embed(token, pos);

        for (li, layer) in self.layers.iter().enumerate() {
            let qkv = Self::plain(layer.qkv.forward(self.row(&x)));
            let attn = cache.layer_step(li, &qkv[d..2 * d], &qkv[..d], &qkv[2 * d..]);

            let projected = Self::plain(layer.out.forward(self.row(&attn)));
            for i in 0..d {
                x[i] += projected[i];
            }

            let ff = Self::plain(layer.ff_in.forward(self.row(&x)));
            let gated: Vec<f64> = (0..f).map(|i| ff[i].max(0.0) * ff[f + i]).collect();
            let back = Self::plain(layer.ff_out.forward(self.row(&gated)));
            for i in 0..d {
                x[i] += back[i];
            }
        }
        x
    }

    /// The token the residual stream decodes to.
    pub fn decode(&self, x: &[f64]) -> usize {
        let logits = Self::plain(self.head.forward(self.row(x)));
        let mut best = 0;
        let mut best_score = f64::NEG_INFINITY;
        for (i, &s) in logits.iter().enumerate() {
            if s > best_score {
                best_score = s;
                best = i;
            }
        }
        best
    }
}
