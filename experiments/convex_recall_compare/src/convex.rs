//! Ordinary answer cross-entropy with directly trained convex parameters.
//!
//! Matching is exactly the forward/domain of FlatSpectrum.ConvexRecall (0fcc0d1b).
//! Copy adapts its fixed value-copying mechanism to two arithmetic input tokens;
//! Table is a separate saturated linear control, not the proved recall model.

use crate::data::{Dataset, Sample};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Matching,
    Copy,
    Table,
}

pub struct Convex {
    pub kind: Kind,
    pub p: Vec<f64>,
    pub keys: usize,
    pub vocab: usize,
    pub lr: f64,
}

/// Euclidean projection onto the closed probability simplex.
pub fn project_simplex(row: &mut [f64]) {
    let mut sorted = row.to_vec();
    sorted.sort_by(|a, b| b.total_cmp(a));
    let (mut sum, mut threshold) = (0.0, 0.0);
    for (j, &v) in sorted.iter().enumerate() {
        sum += v;
        let t = (sum - 1.0) / (j + 1) as f64;
        if v > t {
            threshold = t;
        }
    }
    for x in row {
        *x = (*x - threshold).max(0.0);
    }
}

/// Stable ordinary CE and its derivative in the supplied logits.
pub fn ce_gradient(z: &[f64], target: usize) -> (f64, Vec<f64>) {
    let max = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut grad: Vec<_> = z.iter().map(|x| (x - max).exp()).collect();
    let sum: f64 = grad.iter().sum();
    for g in &mut grad {
        *g /= sum;
    }
    grad[target] -= 1.0;
    (max + sum.ln() - z[target], grad)
}

impl Convex {
    pub fn new(kind: Kind, data: &Dataset) -> Self {
        let p = match kind {
            Kind::Matching => vec![1.0 / data.keys as f64; data.keys * data.keys],
            Kind::Copy => vec![0.5; data.p * 2],
            Kind::Table => vec![0.0; data.p * (data.p - 1) * data.vocab],
        };
        Self { kind, p, keys: data.keys, vocab: data.vocab, lr: 1.0 }
    }

    pub fn logits(&self, sample: &Sample) -> Vec<f64> {
        let mut out = vec![0.0; sample.targets.len() * self.vocab];
        match self.kind {
            Kind::Matching => {
                for (i, &q) in sample.queries.iter().enumerate() {
                    for (k, &v) in sample.values.iter().enumerate() {
                        out[i * self.vocab + self.keys + v] += self.p[q * self.keys + k];
                    }
                }
            }
            Kind::Copy => {
                let q = sample.tokens[0];
                out[sample.tokens[0]] += self.p[2 * q];
                out[sample.tokens[2]] += self.p[2 * q + 1];
            }
            Kind::Table => {
                out.copy_from_slice(&self.p[sample.id * self.vocab..][..self.vocab]);
            }
        }
        out
    }

    pub fn loss_gradient(&self, samples: &[&Sample]) -> (f64, Vec<f64>) {
        let mut gradient = vec![0.0; self.p.len()];
        let count: usize = samples.iter().map(|s| s.targets.len()).sum();
        let mut loss = 0.0;
        for &sample in samples {
            let logits = self.logits(sample);
            for (i, &target) in sample.targets.iter().enumerate() {
                let (ce, dz) = ce_gradient(&logits[i * self.vocab..][..self.vocab], target);
                loss += ce;
                match self.kind {
                    Kind::Matching => {
                        let q = sample.queries[i];
                        for (k, &v) in sample.values.iter().enumerate() {
                            gradient[q * self.keys + k] += dz[self.keys + v];
                        }
                    }
                    Kind::Copy => {
                        let q = sample.tokens[0];
                        gradient[2 * q] += dz[sample.tokens[0]];
                        gradient[2 * q + 1] += dz[sample.tokens[2]];
                    }
                    Kind::Table => {
                        for (g, d) in gradient[sample.id * self.vocab..][..self.vocab].iter_mut().zip(dz) {
                            *g += d;
                        }
                    }
                }
            }
        }
        for g in &mut gradient {
            *g /= count as f64;
        }
        (loss / count as f64, gradient)
    }

    pub fn step(&mut self, samples: &[&Sample]) {
        if self.kind == Kind::Table {
            // Rows not present in the batch have exactly zero derivative.
            // Accumulate repeated rows before changing any parameters.
            let mut rows = std::collections::BTreeMap::<usize, Vec<f64>>::new();
            for &sample in samples {
                let (_, dz) = ce_gradient(&self.logits(sample), sample.targets[0]);
                let row = rows.entry(sample.id).or_insert_with(|| vec![0.0; self.vocab]);
                for (g, d) in row.iter_mut().zip(dz) {
                    *g += d / samples.len() as f64;
                }
            }
            for (id, grad) in rows {
                for (p, g) in self.p[id * self.vocab..][..self.vocab].iter_mut().zip(grad) {
                    *p -= self.lr * g;
                }
            }
            return;
        }
        let (_, gradient) = self.loss_gradient(samples);
        for (p, g) in self.p.iter_mut().zip(gradient) {
            *p -= self.lr * g;
        }
        let width = match self.kind {
            Kind::Matching => self.keys,
            Kind::Copy => 2,
            Kind::Table => return,
        };
        for row in self.p.chunks_exact_mut(width) {
            project_simplex(row);
        }
    }

    pub fn identity_error(&self) -> Option<f64> {
        (self.kind == Kind::Matching).then(|| {
            self.p
                .iter()
                .enumerate()
                .map(|(i, &x)| (x - f64::from(i / self.keys == i % self.keys)).abs())
                .fold(0.0, f64::max)
        })
    }
}
