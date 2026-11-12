//! SB1–SB2 pair-model pilot. Direct complex entries, ordinary CE and convex norm bounds.
//! The frozen generator contains task rules; the model receives only tokens and labels.
pub use convex_stress_minibatch::{Result, data, hash, metrics, random};
pub use num_complex::Complex64 as C;
pub mod linalg;
use serde::{Deserialize, Serialize};

pub fn codes(vocab: usize) -> Vec<C> {
    (0..vocab)
        .map(|c| C::from_polar(1., std::f64::consts::TAU * c as f64 / vocab as f64))
        .collect()
}

pub fn loss(z: C, label: usize, codes: &[C]) -> f64 {
    let logits: Vec<_> = codes.iter().map(|p| (p.conj() * z).re).collect();
    let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    max + logits.iter().map(|x| (x - max).exp()).sum::<f64>().ln() - logits[label]
}

pub fn gradient(z: C, label: usize, codes: &[C]) -> C {
    let logits: Vec<_> = codes.iter().map(|p| (p.conj() * z).re).collect();
    let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let weights: Vec<_> = logits.iter().map(|x| (x - max).exp()).collect();
    let mass: f64 = weights.iter().sum();
    codes
        .iter()
        .zip(weights)
        .map(|(p, w)| *p * (w / mass))
        .sum::<C>()
        - codes[label]
}

pub fn forward_gradient(z: C, label: usize, codes: &[C], h: f64) -> C {
    C::new(
        (loss(z + C::new(h, 0.), label, codes) - loss(z - C::new(h, 0.), label, codes)) / (2. * h),
        (loss(z + C::new(0., h), label, codes) - loss(z - C::new(0., h), label, codes)) / (2. * h),
    )
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Model {
    pub vocab: usize,
    pub radius: f64,
    pub entries: Vec<C>,
}
impl Model {
    pub fn new(vocab: usize, radius: f64) -> Self {
        Self {
            vocab,
            radius,
            entries: vec![C::default(); vocab * vocab],
        }
    }
    /// The last token is the row, the previous token is the column. No task metadata.
    pub fn cell(&self, prefix: &[i64]) -> usize {
        let row = *prefix.last().expect("empty prefix") as usize;
        let col = if prefix.len() > 1 {
            prefix[prefix.len() - 2] as usize
        } else {
            row
        };
        assert!(row < self.vocab && col < self.vocab);
        row + self.vocab * col
    }
    pub fn validate(&self) -> Result<(f64, f64)> {
        let norm = linalg::svd(&self.entries, self.vocab)?.values[0];
        let entry = linalg::max_norm(&self.entries);
        if !self.radius.is_finite()
            || self.radius <= 0.
            || entry > 1. + 1e-8
            || norm > self.radius * (1. + 1e-8)
        {
            return Err("saved model violates spectral/entry bounds".into());
        }
        Ok((entry, norm))
    }
}
impl metrics::Predictor for Model {
    fn vocab(&self) -> usize {
        self.vocab
    }
    fn logits(&self, prefix: &[i64]) -> Vec<f64> {
        let z = self.entries[self.cell(prefix)];
        codes(self.vocab)
            .iter()
            .map(|p| (p.conj() * z).re)
            .collect()
    }
}

#[derive(Clone)]
pub struct Sample {
    pub cell: usize,
    pub label: usize,
}

pub fn samples(model: &Model, examples: &[data::Example]) -> Result<Vec<Sample>> {
    let mut out = vec![];
    let mut seen = std::collections::BTreeSet::new();
    for e in examples {
        let supervised: Vec<_> = e
            .targets
            .iter()
            .enumerate()
            .filter(|(_, y)| **y >= 0)
            .collect();
        if supervised.len() != 1 {
            return Err("pilot requires one supervised prefix per example".into());
        }
        let (position, &label) = supervised[0];
        let cell = model.cell(&e.tokens[..=position]);
        if label as usize >= model.vocab || !seen.insert(cell) {
            return Err("pilot requires unique valid training cells".into());
        }
        out.push(Sample {
            cell,
            label: label as usize,
        });
    }
    Ok(out)
}

pub fn step(
    model: &mut Model,
    samples: &[Sample],
    selected: &[usize],
    forward_difference: Option<f64>,
    eta: f64,
    cap: usize,
    tol: f64,
) -> Result<linalg::Projection> {
    let code = codes(model.vocab);
    let mut point = model.entries.clone();
    for &index in selected {
        let sample = &samples[index];
        let z = model.entries[sample.cell];
        let g = if let Some(h) = forward_difference {
            forward_gradient(z, sample.label, &code, h)
        } else {
            gradient(z, sample.label, &code)
        };
        point[sample.cell] -= eta * g;
    }
    let (x, projection) = linalg::project(&point, model.vocab, model.radius, cap, tol)?;
    model.entries = x;
    Ok(projection)
}

#[cfg(test)]
mod tests;
