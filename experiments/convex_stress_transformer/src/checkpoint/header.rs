//! Small, bounded checkpoint metadata. The binding hashes the caller's frozen
//! case/source/data specification; this module does not define a training budget.
use crate::{Result, adam, engine::Transformer, layout::Dims, memory, mt::Mt};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Progress {
    pub completed_steps: u64,
    pub examples_presented: u64,
    pub targets_presented: u64,
    pub optimizer_seconds: f64,
}
impl Progress {
    pub fn validate(&self) -> Result<()> {
        if !self.optimizer_seconds.is_finite()
            || self.optimizer_seconds < 0.
            || self.targets_presented < self.examples_presented
            || (self.completed_steps == 0
                && (self.examples_presented != 0 || self.targets_presented != 0))
            || (self.completed_steps > 0 && self.examples_presented < self.completed_steps)
        {
            return Err("invalid transformer checkpoint progress".into());
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Header {
    pub version: u32,
    pub binding: String,
    pub dimensions: [usize; 6],
    pub microbatch: usize,
    pub adam_scalars: [f64; 5],
    pub warmup: u64,
    pub progress: Progress,
    pub rng: Vec<u32>,
}
pub(super) fn valid_binding(binding: &str) -> bool {
    binding.len() == 64 && binding.bytes().all(|b| b.is_ascii_hexdigit())
}
impl Header {
    pub fn new(net: &Transformer, rng: &Mt, progress: Progress, binding: &str) -> Result<Self> {
        let d = net.dims;
        let h = net.hyper;
        let out = Self {
            version: 1,
            binding: binding.into(),
            dimensions: [d.vocab, d.d, d.heads, d.layers, d.dff, d.ctx],
            microbatch: net.microbatch,
            adam_scalars: [h.max_lr, h.beta1, h.beta2, h.eps, h.weight_decay],
            warmup: h.warmup,
            progress,
            rng: rng.words(),
        };
        out.validate(binding)?;
        Ok(out)
    }
    pub fn dims(&self) -> Dims {
        let [vocab, d, heads, layers, dff, ctx] = self.dimensions;
        Dims {
            vocab,
            d,
            heads,
            layers,
            dff,
            ctx,
        }
    }
    pub fn hyper(&self) -> adam::Hyper {
        let [max_lr, beta1, beta2, eps, weight_decay] = self.adam_scalars;
        adam::Hyper {
            max_lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            warmup: self.warmup,
        }
    }
    pub fn validate(&self, expected: &str) -> Result<()> {
        if self.version != 1 || !valid_binding(expected) || self.binding != expected {
            return Err("checkpoint version or case/source/data binding mismatch".into());
        }
        memory::check(self.dims(), self.microbatch)?;
        self.progress.validate()?;
        if self.rng.len() != 625 || self.rng[624] > 624 {
            return Err("invalid checkpoint sampling RNG state".into());
        }
        let h = self.hyper();
        if !h.max_lr.is_finite()
            || h.max_lr <= 0.
            || !(0.0..1.0).contains(&h.beta1)
            || !(0.0..1.0).contains(&h.beta2)
            || !h.eps.is_finite()
            || h.eps <= 0.
            || !h.weight_decay.is_finite()
            || h.weight_decay < 0.
        {
            return Err("invalid checkpoint AdamW configuration".into());
        }
        Ok(())
    }
}
