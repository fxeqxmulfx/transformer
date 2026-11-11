//! Conventional two-layer causal softmax transformer, trained with AdamW.
//! Initialization and parameter conventions are those of grok_division at 891e4525.

use crate::adam::{self, Hyper};
use crate::data::{Dataset, Sample};
use crate::layout::Dims;
use crate::model::{Model, Transposed, Work};
use crate::mt::Mt;

pub fn initialize(dims: Dims, seed: u32) -> Vec<f32> {
    let mut rng = Mt::new(1_000_003u32.wrapping_mul(seed.wrapping_add(1)));
    let (d, dff) = (dims.d, dims.dff);
    let mut p = vec![0.0; dims.len()];
    fn uniform(slice: &mut [f32], fan: usize, rng: &mut Mt) {
        for x in slice {
            *x = ((2.0 * rng.double() - 1.0) / (fan as f64).sqrt()) as f32;
        }
    }
    for x in &mut p[..dims.vocab * d] {
        *x = rng.normal() as f32;
    }
    for l in 0..2 {
        let at = dims.block(l);
        uniform(&mut p[at.wq..at.wq + d * d], d, &mut rng);
        uniform(&mut p[at.wkv..at.wkv + 2 * d * d], d, &mut rng);
        uniform(&mut p[at.wo..at.wo + d * d], d, &mut rng);
        p[at.g1..at.g1 + d].fill(1.0);
        uniform(&mut p[at.w1..at.w1 + d * dff], d, &mut rng);
        uniform(&mut p[at.w2..at.w2 + dff * d], dff, &mut rng);
        p[at.g2..at.g2 + d].fill(1.0);
    }
    uniform(&mut p[dims.wout()..], d, &mut rng);
    p
}

pub struct Transformer {
    pub dims: Dims,
    pub p: Vec<f32>,
    pub m: Vec<f32>,
    pub v: Vec<f32>,
    pub hyper: Hyper,
    pe: Vec<f32>,
    wt: Transposed,
    work: Vec<Work>,
    sup: Vec<usize>,
}

impl Transformer {
    pub fn new(data: &Dataset, seed: u32, width: usize, threads: usize, decay: f64) -> Self {
        let dims = Dims { vocab: data.vocab, d: width, heads: 4, layers: 2, dff: width * 4, ctx: data.ctx };
        Self {
            dims,
            p: initialize(dims, seed),
            m: vec![0.0; dims.len()],
            v: vec![0.0; dims.len()],
            hyper: Hyper::source(decay),
            pe: dims.positions(),
            wt: Transposed::new(dims),
            work: (0..threads).map(|_| Work::default()).collect(),
            sup: data.sup.clone(),
        }
    }

    pub fn step(&mut self, samples: &[&Sample], step: u64) {
        self.wt.update(self.dims, &self.p);
        let tokens: Vec<_> = samples.iter().flat_map(|s| s.tokens.iter().copied()).collect();
        let targets: Vec<_> = samples.iter().flat_map(|s| s.targets.iter().copied()).collect();
        let scale = 1.0 / targets.len() as f32;
        let chunk = samples.len().div_ceil(self.work.len());
        let nwork = samples.len().div_ceil(chunk);
        let model = Model { dims: self.dims, p: &self.p, pe: &self.pe, wt: &self.wt, sup: &self.sup };
        std::thread::scope(|scope| {
            for (worker, work) in self.work[..nwork].iter_mut().enumerate() {
                let start = worker * chunk;
                let end = ((worker + 1) * chunk).min(samples.len());
                let tok = &tokens[start * self.dims.ctx..end * self.dims.ctx];
                let tgt = &targets[start * self.sup.len()..end * self.sup.len()];
                let model = &model;
                scope.spawn(move || {
                    model.forward(work, tok);
                    model.backward(work, tok, tgt, scale);
                });
            }
        });
        let mut gradient = vec![0.0; self.p.len()];
        for (i, g) in gradient.iter_mut().enumerate() {
            *g = self.work[..nwork].iter().map(|w| w.grad[i] as f64).sum::<f64>() as f32;
        }
        assert!(gradient.iter().all(|x| x.is_finite()), "nonfinite transformer gradient");
        adam::update(&self.hyper, step, &mut self.p, &mut self.m, &mut self.v, &gradient);
    }

    pub fn logits(&mut self, samples: &[Sample]) -> Vec<f64> {
        let tokens: Vec<_> = samples.iter().flat_map(|s| s.tokens.iter().copied()).collect();
        let model = Model { dims: self.dims, p: &self.p, pe: &self.pe, wt: &self.wt, sup: &self.sup };
        model.forward(&mut self.work[0], &tokens);
        self.work[0].logits.iter().map(|&x| x as f64).collect()
    }
}
