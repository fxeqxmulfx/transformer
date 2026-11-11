//! The update of openai/grok `grok/training.py` at 3d64b1d8: `CustomAdamW` in its `to_zero`
//! form without noise or AMSGrad, at the learning rate of `_scheduler_lr` without annealing.
//! Every parameter is in the one group, so all of them decay, normalization gains and biases
//! too. Scalars are worked out in doubles and rounded to `f32` where PyTorch hands a Python
//! float to an `f32` tensor operation; the tensor operations are `f32`, in the source's order.

/// The optimizer's constants.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Hyper {
    /// The learning rate after warmup, `max_lr`.
    pub max_lr: f64,
    /// Updates of linear warmup, `warmup_steps`.
    pub warmup: u64,
    /// The first moment's decay.
    pub beta1: f64,
    /// The second moment's decay.
    pub beta2: f64,
    /// The constant added to the root of the corrected second moment.
    pub eps: f64,
    /// The decoupled weight decay; 0 leaves the parameters alone.
    pub weight_decay: f64,
}

impl Hyper {
    /// The source's constants at the weight decay given: learning rate 0.001, 10 updates of
    /// warmup, betas (0.9, 0.98), epsilon 1e-8.
    pub fn source(weight_decay: f64) -> Hyper {
        Hyper { max_lr: 1e-3, warmup: 10, beta1: 0.9, beta2: 0.98, eps: 1e-8, weight_decay }
    }

    /// The learning rate of update `t`, counted from 1. `LambdaLR` has stepped `t - 1` times
    /// when the optimizer makes its update `t`, so the first update has rate 0 and update
    /// `warmup + 1` is the first at `max_lr`.
    pub fn lr(&self, t: u64) -> f64 {
        let step = (t - 1) as f64;
        if t - 1 <= self.warmup { step / self.warmup.max(1) as f64 * self.max_lr } else { self.max_lr }
    }
}

/// Update `t`, counted from 1, of the parameters `p` with moments `m` and `v`, at the gradient
/// `g`: the decay `p *= 1 - lr wd` first, then the moments, then
/// `p -= lr / (1 - beta1^t) * m / (sqrt(v) / sqrt(1 - beta2^t) + eps)`.
pub fn update(h: &Hyper, t: u64, p: &mut [f32], m: &mut [f32], v: &mut [f32], g: &[f32]) {
    assert!(m.len() == p.len() && v.len() == p.len() && g.len() == p.len(), "the tensors differ in length");
    let lr = h.lr(t);
    let decay = (1.0 - lr * h.weight_decay) as f32;
    let (b1, c1) = (h.beta1 as f32, (1.0 - h.beta1) as f32);
    let (b2, c2) = (h.beta2 as f32, (1.0 - h.beta2) as f32);
    let correction1 = 1.0 - h.beta1.powf(t as f64);
    let correction2 = 1.0 - h.beta2.powf(t as f64);
    let (root2, eps) = (correction2.sqrt() as f32, h.eps as f32);
    let step = (-(lr / correction1)) as f32;
    if h.weight_decay > 0.0 {
        p.iter_mut().for_each(|p| *p *= decay);
    }
    for (((p, m), v), &g) in p.iter_mut().zip(m.iter_mut()).zip(v.iter_mut()).zip(g) {
        *m = *m * b1 + g * c1;
        *v = *v * b2 + (c2 * g) * g;
        let denom = v.sqrt() / root2 + eps;
        *p += step * (*m / denom);
    }
}
