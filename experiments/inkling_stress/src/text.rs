//! Text-only, task-scaled Inkling 2 Mini. Four blocks with two layers each.
//! See the experiment README for the pinned release, dimensions and precision.
use tch::{Kind, Tensor, nn};

#[path = "text/attention.rs"]
mod attention;
#[path = "text/block.rs"]
mod block;
use block::Layer;

pub const MODEL: &str = "inkling2-text-8layer-8expert-top2-3pass-tied-fp32-v1";
pub const PASSES: usize = 3;
const GROUP: usize = 2;
const BEGIN: usize = GROUP;
const END: usize = 3 * GROUP;
const EXPERTS: i64 = 8;
const TOP: i64 = 2;
const EPS: f64 = 1e-6;

fn weights(p: &nn::Path, name: &str, shape: &[i64]) -> Tensor {
    p.randn(name, shape, 0., 0.02)
}
fn rms(x: &Tensor) -> Tensor {
    x * (x.square().mean_dim(-1, true, Kind::Float) + EPS).rsqrt()
}
fn softcap(x: &Tensor, bound: f64) -> Tensor {
    (x / bound).tanh() * bound
}
/// Attention over earlier layer states at the same time position.
fn read_states(states: &[Tensor], partial: &Tensor, queries: &Tensor) -> Tensor {
    let mut values: Vec<&Tensor> = states.iter().collect();
    values.push(partial);
    let values = rms(&Tensor::stack(&values, -2));
    let scale = (partial.size()[2] as f64).sqrt();
    (queries.matmul(&values.transpose(-2, -1)) / scale)
        .softmax(-1, Kind::Float)
        .matmul(&values)
}

struct Gate {
    down: Tensor,
    forget: Tensor,
    admit: Tensor,
}
impl Gate {
    fn new(p: &nn::Path, width: i64) -> Self {
        let rank = (width / 16).max(1);
        Self {
            down: weights(p, "down", &[2 * width, rank]),
            forget: p.zeros("forget", &[rank, width]),
            admit: p.zeros("admit", &[rank, width]),
        }
    }
    fn forward(&self, x: &Tensor, m: &Tensor) -> Tensor {
        let h = Tensor::cat(&[rms(x), rms(m)], -1).matmul(&self.down);
        (1. - h.matmul(&self.forget).tanh()) * x + h.matmul(&self.admit).tanh() * m
    }
}

pub struct Model {
    pub embedding: Tensor,
    embed_gain: Tensor,
    layers: Vec<Layer>,
    final_query: Tensor,
    final_gain: Tensor,
}
impl Model {
    pub fn new(p: &nn::Path, vocab: i64, width: i64, heads: i64) -> Self {
        Self {
            embedding: weights(p, "embedding", &[vocab, width]),
            embed_gain: p.ones("embed_gain", &[width]),
            layers: (0..4 * GROUP)
                .map(|i| Layer::new(&(p / format!("layer{i}")), i, width, heads))
                .collect(),
            final_query: p.zeros("final_query", &[1, width]),
            final_gain: p.ones("final_gain", &[width]),
        }
    }
    pub fn inner(&self, embeddings: &Tensor, passes: usize) -> Tensor {
        assert!(passes > 0);
        let mut prelude = rms(embeddings) * &self.embed_gain;
        let mut pre_blocks = Vec::new();
        for layer in &self.layers[..BEGIN] {
            prelude = layer.apply(prelude, &mut pre_blocks, None);
        }
        let mut h = prelude.shallow_clone();
        let mut blocks: Vec<_> = pre_blocks.iter().map(Tensor::shallow_clone).collect();
        for layer in &self.layers[BEGIN..END] {
            h = layer.apply(h, &mut blocks, None);
        }
        for _ in 1..passes {
            let previous = h;
            // Drop the first core boundary, which holds the saved prelude.
            let previous_blocks = blocks.split_off(pre_blocks.len() + 1);
            blocks.truncate(pre_blocks.len());
            h = prelude.shallow_clone();
            for layer in &self.layers[BEGIN..END] {
                let reader = layer
                    .reader
                    .as_ref()
                    .expect("middle layer owns a message reader");
                let messages = read_states(&previous_blocks, &previous, &reader.0);
                h = layer.apply(h, &mut blocks, Some(&messages));
            }
        }
        for layer in &self.layers[END..] {
            h = layer.apply(h, &mut blocks, None);
        }
        rms(&read_states(&blocks, &h, &self.final_query).squeeze_dim(-2)) * &self.final_gain
    }
    pub fn logits(&self, ids: &Tensor) -> Tensor {
        let embeddings = Tensor::embedding(&self.embedding, ids, -1, false, false);
        let hidden = self.inner(&embeddings, PASSES);
        // Tied suite vocabulary, with the release's monotone final logit cap.
        softcap(&hidden.matmul(&self.embedding.transpose(0, 1)), 30.)
    }
}
