//! Streamed, token-weighted ordinary CE on variable-length, arbitrarily masked
//! original examples. Microbatches change accumulation storage, not Adam's batch.
//! Initialization follows convex_recall_compare at da0714aa for every block.
use crate::{Result, adam, data::Example, layout::Dims, memory, model, mt::Mt};
use std::{cell::RefCell, collections::BTreeMap};

pub fn initialize(dims: Dims, seed: u32) -> Result<Vec<f32>> {
    memory::check(dims, 1)?;
    let mut rng = Mt::new(1_000_003u32.wrapping_mul(seed.wrapping_add(1)));
    let (d, dff) = (dims.d, dims.dff);
    let mut p = vec![0.; dims.len()];
    fn uniform(slice: &mut [f32], fan: usize, rng: &mut Mt) {
        for x in slice {
            *x = ((2. * rng.double() - 1.) / (fan as f64).sqrt()) as f32;
        }
    }
    for x in &mut p[..dims.vocab * d] {
        *x = rng.normal() as f32;
    }
    for l in 0..dims.layers {
        let at = dims.block(l);
        uniform(&mut p[at.wq..at.wq + d * d], d, &mut rng);
        uniform(&mut p[at.wkv..at.wkv + 2 * d * d], d, &mut rng);
        uniform(&mut p[at.wo..at.wo + d * d], d, &mut rng);
        p[at.g1..at.g1 + d].fill(1.);
        uniform(&mut p[at.w1..at.w1 + d * dff], d, &mut rng);
        uniform(&mut p[at.w2..at.w2 + dff * d], dff, &mut rng);
        p[at.g2..at.g2 + d].fill(1.);
    }
    uniform(&mut p[dims.wout()..], d, &mut rng);
    Ok(p)
}

pub fn validate_example(dims: Dims, e: &Example) -> Result<usize> {
    if e.tokens.is_empty()
        || e.tokens.len() > dims.ctx
        || e.tokens.len() != e.targets.len()
        || e.tokens.iter().any(|&x| x < 0 || x as usize >= dims.vocab)
        || e.targets
            .iter()
            .any(|&x| x != -100 && (x < 0 || x as usize >= dims.vocab))
    {
        return Err("invalid transformer training tokens or target mask".into());
    }
    let n = e.targets.iter().filter(|&&x| x >= 0).count();
    if n == 0 {
        return Err("example has no supervised target".into());
    }
    Ok(n)
}

pub struct Transformer {
    pub dims: Dims,
    pub p: Vec<f32>,
    pub m: Vec<f32>,
    pub v: Vec<f32>,
    pub hyper: adam::Hyper,
    pub microbatch: usize,
    pe: Vec<f32>,
    wt: model::Transposed,
    work: model::Work,
}

impl Transformer {
    pub fn new(dims: Dims, seed: u32, microbatch: usize, hyper: adam::Hyper) -> Result<Self> {
        memory::check(dims, microbatch)?;
        if !hyper.max_lr.is_finite()
            || hyper.max_lr <= 0.
            || !(0.0..1.0).contains(&hyper.beta1)
            || !(0.0..1.0).contains(&hyper.beta2)
            || !hyper.eps.is_finite()
            || hyper.eps <= 0.
            || !hyper.weight_decay.is_finite()
            || hyper.weight_decay < 0.
        {
            return Err("invalid AdamW configuration".into());
        }
        Ok(Self {
            dims,
            p: initialize(dims, seed)?,
            m: vec![0.; dims.len()],
            v: vec![0.; dims.len()],
            hyper,
            microbatch,
            pe: dims.positions(),
            wt: model::Transposed::new(dims),
            work: model::Work::default(),
        })
    }

    pub fn loss_gradient(&mut self, examples: &[&Example]) -> Result<(f64, Vec<f32>, usize)> {
        memory::check(self.dims, self.microbatch)?;
        if examples.is_empty() {
            return Err("empty optimizer batch".into());
        }
        let targets = examples.iter().try_fold(0usize, |n, e| -> Result<usize> {
            n.checked_add(validate_example(self.dims, e)?)
                .ok_or_else(|| "target count overflow".into())
        })?;
        if self.p.len() != self.dims.len() || self.p.iter().any(|x| !x.is_finite()) {
            return Err("invalid transformer parameters".into());
        }
        self.wt.update(self.dims, &self.p);
        let mut gradient = vec![0f64; self.p.len()];
        let mut loss = 0.;
        // Only one bounded chunk's grouping and token copies are live. The
        // complete optimizer batch is never materialized as an activation tensor.
        for chunk in examples.chunks(self.microbatch) {
            let mut groups: BTreeMap<(usize, Vec<usize>), Vec<&Example>> = BTreeMap::new();
            for &e in chunk {
                let sup = e
                    .targets
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &y)| (y >= 0).then_some(i))
                    .collect();
                groups.entry((e.tokens.len(), sup)).or_default().push(e);
            }
            for ((ctx, sup), group) in groups {
                let tokens: Vec<_> = group
                    .iter()
                    .flat_map(|e| e.tokens.iter().map(|&x| x as usize))
                    .collect();
                let labels: Vec<_> = group
                    .iter()
                    .flat_map(|e| sup.iter().map(|&i| e.targets[i] as usize))
                    .collect();
                let net = model::Model {
                    dims: Dims { ctx, ..self.dims },
                    p: &self.p,
                    pe: &self.pe,
                    wt: &self.wt,
                    sup: &sup,
                };
                net.forward(&mut self.work, &tokens);
                loss += net
                    .backward(&mut self.work, &tokens, &labels, 1. / targets as f32)
                    .iter()
                    .sum::<f64>();
                for (to, &from) in gradient.iter_mut().zip(&self.work.grad) {
                    *to += from as f64;
                }
            }
        }
        let gradient: Vec<f32> = gradient.into_iter().map(|x| x as f32).collect();
        if !loss.is_finite() || gradient.iter().any(|x| !x.is_finite()) {
            return Err("nonfinite transformer CE or gradient".into());
        }
        Ok((loss / targets as f64, gradient, targets))
    }

    pub fn step(&mut self, examples: &[&Example], completed_step: u64) -> Result<(f64, usize)> {
        if completed_step == 0 {
            return Err("AdamW updates start at one".into());
        }
        let (loss, gradient, targets) = self.loss_gradient(examples)?;
        adam::update(
            &self.hyper,
            completed_step,
            &mut self.p,
            &mut self.m,
            &mut self.v,
            &gradient,
        );
        if self
            .p
            .iter()
            .chain(&self.m)
            .chain(&self.v)
            .any(|x| !x.is_finite())
        {
            return Err("nonfinite AdamW state".into());
        }
        Ok((loss, targets))
    }

    /// One causal forward pass for any ordered subset of a single input's positions.
    pub fn logits(&mut self, tokens: &[i64], sup: &[usize]) -> Result<Vec<f64>> {
        if tokens.is_empty()
            || tokens.len() > self.dims.ctx
            || sup.is_empty()
            || tokens
                .iter()
                .any(|&x| x < 0 || x as usize >= self.dims.vocab)
            || sup.last().is_some_and(|&x| x >= tokens.len())
            || sup.windows(2).any(|w| w[0] >= w[1])
        {
            return Err("invalid causal prediction request".into());
        }
        let tokens: Vec<_> = tokens.iter().map(|&x| x as usize).collect();
        let net = model::Model {
            dims: Dims {
                ctx: tokens.len(),
                ..self.dims
            },
            p: &self.p,
            pe: &self.pe,
            wt: &self.wt,
            sup,
        };
        net.forward(&mut self.work, &tokens);
        if self.work.logits.iter().any(|x| !x.is_finite()) {
            return Err("nonfinite logits".into());
        }
        Ok(self.work.logits.iter().map(|&x| x as f64).collect())
    }
}

/// Uses the benchmark's unchanged metrics, including greedy autoregressive
/// generation and residue pooling. Inference receives only the visible prefix.
pub struct Predictor<'a>(pub RefCell<&'a mut Transformer>);
impl crate::metrics::Predictor for Predictor<'_> {
    fn vocab(&self) -> usize {
        self.0.borrow().dims.vocab
    }
    fn logits(&self, prefix: &[i64]) -> Vec<f64> {
        self.0
            .borrow_mut()
            .logits(
                prefix,
                &[prefix.len().checked_sub(1).expect("nonempty prefix")],
            )
            .expect("validated original data must produce finite causal logits")
    }
}
