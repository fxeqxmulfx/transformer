//! CD8's learned tensor, ordinary CE and per-class L1 budget. Input features
//! remain fixed. All original supervised positions and repeated examples stay
//! in the empirical loss. The forward initializer reads training labels once;
//! its complete cost must be counted for every mode that uses it.
use crate::{Result, data, features::Bank, metrics::Predictor};
use serde::{Deserialize, Serialize};

pub const BUFFER_LIMIT: usize = 2 * 1024 * 1024 * 1024;

pub struct Cache {
    pub active: Vec<u32>,
    pub labels: Vec<usize>,
    pub example_offsets: Vec<usize>,
    pub width: usize,
    pub dimension: usize,
    pub vocab: usize,
}
impl Cache {
    pub fn build(bank: &Bank, examples: &[data::Example]) -> Result<Self> {
        let n = examples
            .iter()
            .map(|e| e.targets.iter().filter(|t| **t >= 0).count())
            .sum::<usize>();
        if n == 0 || bank.allocation_bytes(n)? > BUFFER_LIMIT {
            return Err("empty training set or explicit buffers exceed 2 GiB".into());
        }
        let width = 2 * bank.paths();
        let mut active = Vec::with_capacity(n * width);
        let mut labels = Vec::with_capacity(n);
        let mut example_offsets = vec![0];
        for e in examples {
            if e.targets.len() != e.tokens.len() {
                return Err("input/target length mismatch".into());
            }
            for (position, &target) in e.targets.iter().enumerate() {
                if target < 0 {
                    continue;
                }
                if target as usize >= bank.vocab {
                    return Err("invalid target".into());
                }
                for k in bank.encode(&e.tokens[..=position])? {
                    active.push(u32::try_from(k)?);
                }
                labels.push(target as usize);
            }
            if labels.len() == *example_offsets.last().unwrap() {
                return Err("example without supervision".into());
            }
            example_offsets.push(labels.len());
        }
        Ok(Self {
            active,
            labels,
            example_offsets,
            width,
            dimension: bank.dimension(),
            vocab: bank.vocab,
        })
    }
    pub fn features(&self, i: usize) -> &[u32] {
        &self.active[i * self.width..(i + 1) * self.width]
    }
    pub fn full_order(&self) -> Vec<usize> {
        (0..self.example_offsets.len() - 1).collect()
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Head {
    pub bank: Bank,
    pub radius: f64,
    pub coefficients: Vec<f64>,
}
impl Head {
    pub fn zero(bank: Bank, radius: f64) -> Result<Self> {
        bank.validate()?;
        if !radius.is_finite() || radius <= 0. || bank.allocation_bytes(0)? > BUFFER_LIMIT {
            return Err("invalid head radius or allocation".into());
        }
        Ok(Self {
            coefficients: vec![0.; bank.dimension() * bank.vocab],
            bank,
            radius,
        })
    }
    pub fn validate(&self) -> Result<f64> {
        self.bank.validate()?;
        if !self.radius.is_finite()
            || self.radius <= 0.
            || self.coefficients.len() != self.bank.dimension() * self.bank.vocab
            || self.coefficients.iter().any(|x| !x.is_finite())
        {
            return Err("invalid output tensor".into());
        }
        let mut sums = vec![0.; self.bank.vocab];
        for row in self.coefficients.chunks_exact(self.bank.vocab) {
            for (sum, x) in sums.iter_mut().zip(row) {
                *sum += x.abs();
            }
        }
        let largest = sums.into_iter().fold(0., f64::max);
        if largest > self.radius * (1. + 1e-10) {
            return Err("output tensor violates per-class L1 budget".into());
        }
        Ok(largest)
    }
    pub fn encoded_logits(&self, features: &[u32], out: &mut [f64]) {
        out.fill(0.);
        for &k in features {
            let start = k as usize * self.bank.vocab;
            for (z, w) in out
                .iter_mut()
                .zip(&self.coefficients[start..start + self.bank.vocab])
            {
                *z += w;
            }
        }
    }
    pub fn project(&mut self) {
        let v = self.bank.vocab;
        let mut scratch = Vec::with_capacity(self.bank.dimension());
        for c in 0..v {
            scratch.clear();
            scratch.extend(self.coefficients.iter().skip(c).step_by(v).map(|x| x.abs()));
            if scratch.iter().sum::<f64>() <= self.radius {
                continue;
            }
            scratch.sort_unstable_by(|a, b| b.total_cmp(a));
            let (mut sum, mut threshold) = (0., 0.);
            for (j, &x) in scratch.iter().enumerate() {
                sum += x;
                let next = (sum - self.radius) / (j + 1) as f64;
                if x > next {
                    threshold = next;
                }
            }
            for x in self.coefficients.iter_mut().skip(c).step_by(v) {
                *x = x.signum() * (x.abs() - threshold).max(0.);
            }
        }
    }
}
impl Predictor for Head {
    fn vocab(&self) -> usize {
        self.bank.vocab
    }
    fn logits(&self, prefix: &[i64]) -> Vec<f64> {
        let features = self.bank.encode(prefix).expect("invalid evaluation prefix");
        let mut result = vec![0.; self.bank.vocab];
        for k in features {
            for (z, w) in result
                .iter_mut()
                .zip(&self.coefficients[k * self.bank.vocab..(k + 1) * self.bank.vocab])
            {
                *z += w;
            }
        }
        result
    }
}

/// Ordinary CE and its derivative in logits, using a log-sum-exp shift.
/// `exp_m1` is not needed: this expression also handles nonseparable rows.
pub fn ce_gradient(logits: &[f64], target: usize, residual: &mut [f64]) -> f64 {
    let top = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mass: f64 = logits.iter().map(|z| (z - top).exp()).sum();
    for (c, (r, z)) in residual.iter_mut().zip(logits).enumerate() {
        *r = (z - top).exp() / mass - f64::from(c == target);
    }
    top - logits[target] + mass.ln()
}

/// A forward-only loss evaluation: no loss derivative is formed.
pub fn cross_entropy(logits: &[f64], target: usize) -> f64 {
    let top = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    top - logits[target] + logits.iter().map(|z| (z - top).exp()).sum::<f64>().ln()
}

/// Average over every supervised prefix in the selected original examples.
/// Passing the original order computes the full empirical objective; minibatch
/// order refers to examples, including their complete supervision and tails.
pub fn objective(
    model: &Head,
    cache: &Cache,
    examples: &[usize],
    mut gradient: Option<&mut [f64]>,
) -> Result<(f64, usize)> {
    if examples.is_empty()
        || model.bank.dimension() != cache.dimension
        || model.bank.vocab != cache.vocab
        || gradient
            .as_ref()
            .is_some_and(|g| g.len() != model.coefficients.len())
    {
        return Err("incompatible objective inputs".into());
    }
    if let Some(g) = gradient.as_deref_mut() {
        g.fill(0.);
    }
    let (mut total, mut count) = (0., 0usize);
    let mut logits = vec![0.; cache.vocab];
    let mut residual = vec![0.; cache.vocab];
    for &e in examples {
        if e >= cache.example_offsets.len() - 1 {
            return Err("invalid example index".into());
        }
        for i in cache.example_offsets[e]..cache.example_offsets[e + 1] {
            model.encoded_logits(cache.features(i), &mut logits);
            total += if gradient.is_some() {
                ce_gradient(&logits, cache.labels[i], &mut residual)
            } else {
                cross_entropy(&logits, cache.labels[i])
            };
            count += 1;
            if let Some(g) = gradient.as_deref_mut() {
                for &k in cache.features(i) {
                    let start = k as usize * cache.vocab;
                    for (x, r) in g[start..start + cache.vocab].iter_mut().zip(&residual) {
                        *x += r;
                    }
                }
            }
        }
    }
    if count == 0 || !total.is_finite() {
        return Err("nonfinite or empty CE".into());
    }
    if let Some(g) = gradient {
        for x in g {
            *x /= count as f64;
        }
    }
    Ok((total / count as f64, count))
}

/// Numerical full-gradient linear-oracle diagnostic for a product of L1 balls.
/// It is meaningful for the full empirical gradient, not a minibatch estimate.
pub fn linear_gap(model: &Head, gradient: &[f64]) -> Result<f64> {
    if gradient.len() != model.coefficients.len() || gradient.iter().any(|x| !x.is_finite()) {
        return Err("invalid gradient for oracle".into());
    }
    let mut maxima = vec![0f64; model.bank.vocab];
    let dot: f64 = gradient
        .iter()
        .zip(&model.coefficients)
        .map(|(g, w)| g * w)
        .sum();
    for row in gradient.chunks_exact(model.bank.vocab) {
        for (m, g) in maxima.iter_mut().zip(row) {
            *m = m.max(g.abs());
        }
    }
    Ok(dot + model.radius * maxima.iter().sum::<f64>())
}

pub fn gradient_step(model: &mut Head, gradient: &[f64], learning_rate: f64) -> Result<()> {
    if gradient.len() != model.coefficients.len()
        || !learning_rate.is_finite()
        || learning_rate <= 0.
        || gradient.iter().any(|x| !x.is_finite())
    {
        return Err("invalid gradient step".into());
    }
    for (w, g) in model.coefficients.iter_mut().zip(gradient) {
        *w -= learning_rate * g;
    }
    model.project();
    model.validate()?;
    Ok(())
}

/// Forward-only empirical bucket proposals. Each of the 2*paths channels uses
/// exactly one feature per observation. Label counts produce log frequencies
/// floored at exp(-24); subtracting the row minimum preserves softmax. Scale
/// each whole candidate into the budget and choose only its original training
/// CE, with earliest-channel tie breaking. No held-out input or gradient enters.
pub fn forward_initialize(model: &mut Head, cache: &Cache, counts: &mut [f64]) -> Result<usize> {
    if counts.len() != model.coefficients.len()
        || model.bank.dimension() != cache.dimension
        || model.bank.vocab != cache.vocab
    {
        return Err("incompatible forward initializer".into());
    }
    counts.fill(0.);
    let v = cache.vocab;
    for i in 0..cache.labels.len() {
        for &k in cache.features(i) {
            counts[k as usize * v + cache.labels[i]] += 1.;
        }
    }
    let stride = model.bank.stride();
    let mut best = (f64::INFINITY, 0usize, 1.);
    let mut scores = vec![0.; v];
    for channel in 0..cache.width {
        let path = channel / 2;
        let (begin, end) = if channel % 2 == 0 {
            (0, v + 1)
        } else {
            (v + 1, stride)
        };
        let mut norms = vec![0.; v];
        for feature in begin..end {
            let k = path * stride + feature;
            bucket_scores(&counts[k * v..(k + 1) * v], &mut scores);
            for (sum, z) in norms.iter_mut().zip(&scores) {
                *sum += z.abs();
            }
        }
        let maximum = norms.into_iter().fold(0., f64::max);
        let scale = if maximum > 0. {
            (model.radius / maximum).min(1.)
        } else {
            1.
        };
        let mut loss = 0.;
        for feature in begin..end {
            let k = path * stride + feature;
            let row = &counts[k * v..(k + 1) * v];
            bucket_scores(row, &mut scores);
            for z in &mut scores {
                *z *= scale;
            }
            let top = cross_entropy(&scores, 0) + scores[0];
            for (count, z) in row.iter().zip(&scores) {
                loss += count * (top - z);
            }
        }
        loss /= cache.labels.len() as f64;
        if loss < best.0 {
            best = (loss, channel, scale);
        }
    }
    model.coefficients.fill(0.);
    let path = best.1 / 2;
    let (begin, end) = if best.1 % 2 == 0 {
        (0, v + 1)
    } else {
        (v + 1, stride)
    };
    for feature in begin..end {
        let k = path * stride + feature;
        bucket_scores(&counts[k * v..(k + 1) * v], &mut scores);
        for (w, z) in model.coefficients[k * v..(k + 1) * v]
            .iter_mut()
            .zip(&scores)
        {
            *w = best.2 * z;
        }
    }
    model.validate()?;
    Ok(best.1)
}

fn bucket_scores(counts: &[f64], out: &mut [f64]) {
    let total: f64 = counts.iter().sum();
    if total == 0. {
        out.fill(0.);
        return;
    }
    for (z, &n) in out.iter_mut().zip(counts) {
        *z = if n == 0. {
            -24.
        } else {
            (n / total).ln().max(-24.)
        };
    }
    let minimum = out.iter().copied().fold(f64::INFINITY, f64::min);
    for z in out {
        *z -= minimum;
    }
}

#[cfg(test)]
mod tests;
