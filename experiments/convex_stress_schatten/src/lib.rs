//! CM5–CM7: operator and nuclear budgets for common complex class moments.
//! Ported from coupled-moments 2de90819; CE/FFT/data/batching are retained, while
//! projection and saved-model validation additionally enforce the nuclear budget.
//! Lean proves convexity, a rank-one CE recovery case and a convex-hull certificate;
//! numerical iterations have no finite-step convergence theorem.
pub use convex_stress_moments::{C, Result, data, fft, hash, linalg, metrics, random};
pub mod projection;
pub mod solver;
use serde::{Deserialize, Serialize};

pub fn loss(logits: &[f64], label: usize) -> f64 {
    let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    max + logits.iter().map(|x| (x - max).exp()).sum::<f64>().ln() - logits[label]
}
pub fn gradient(logits: &[f64], label: usize, out: &mut [f64]) {
    assert_eq!(logits.len(), out.len());
    let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mass: f64 = logits.iter().map(|x| (x - max).exp()).sum();
    for (c, (g, &x)) in out.iter_mut().zip(logits).enumerate() {
        *g = (x - max).exp() / mass - f64::from(c == label);
    }
}
pub fn ideal_loss(vocab: usize) -> f64 {
    (std::f64::consts::E + vocab as f64 - 1.).ln() - 1.
}

/// CM7 certificate search: use only training CE. Prefer the first prospectively
/// ordered candidate meeting the ideal-loss tolerance, else the lowest training
/// CE (the latter has no global hull-optimality certificate).
pub fn choose_by_train(losses: &[f64], ideal: f64, tolerance: f64) -> Result<usize> {
    if losses.is_empty()
        || losses.iter().any(|x| !x.is_finite() || *x < ideal - 1e-10)
        || !ideal.is_finite()
        || !tolerance.is_finite()
        || tolerance < 0.
    {
        return Err("invalid training-only selection inputs".into());
    }
    Ok(losses
        .iter()
        .position(|x| x - ideal <= tolerance)
        .unwrap_or_else(|| {
            (0..losses.len())
                .min_by(|&a, &b| losses[a].total_cmp(&losses[b]).then(a.cmp(&b)))
                .unwrap()
        }))
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Model {
    pub vocab: usize,
    pub radius: f64,
    pub nuclear_budget: f64,
    pub probabilities: Vec<f64>,
}
impl Model {
    pub fn new(vocab: usize, radius: f64, nuclear_budget: f64) -> Result<Self> {
        if !(2..=256).contains(&vocab)
            || !radius.is_finite()
            || radius <= 0.
            || !nuclear_budget.is_finite()
            || nuclear_budget <= 0.
        {
            return Err("invalid model dimensions or radius".into());
        }
        Ok(Self {
            vocab,
            radius,
            nuclear_budget,
            probabilities: vec![1. / vocab as f64; vocab * vocab * vocab],
        })
    }
    /// The last token is the row and the previous token the column. Earlier
    /// prefix content is ignored in this pilot; no arithmetic rule is evaluated.
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
    pub fn validate_shape(&self) -> Result<()> {
        if !(2..=256).contains(&self.vocab)
            || !self.radius.is_finite()
            || self.radius <= 0.
            || !self.nuclear_budget.is_finite()
            || self.nuclear_budget <= 0.
            || self.probabilities.len() != self.vocab * self.vocab * self.vocab
            || self
                .probabilities
                .iter()
                .any(|p| !p.is_finite() || *p < -1e-10 || *p > 1. + 1e-10)
            || self
                .probabilities
                .chunks_exact(self.vocab)
                .any(|p| (p.iter().sum::<f64>() - 1.).abs() > 1e-10)
        {
            return Err("invalid saved probability table".into());
        }
        Ok(())
    }
    pub fn validate(
        &self,
        projector: &mut projection::Projector,
    ) -> Result<Vec<projection::Norms>> {
        self.validate_shape()?;
        let norms = projector.norms(&self.probabilities)?;
        if norms.iter().any(|s| {
            s.operator > self.radius * (1. + 1e-9) || s.nuclear > self.nuclear_budget * (1. + 1e-9)
        }) {
            return Err("saved model violates a moment operator/nuclear bound".into());
        }
        Ok(norms)
    }
}
impl metrics::Predictor for Model {
    fn vocab(&self) -> usize {
        self.vocab
    }
    fn logits(&self, prefix: &[i64]) -> Vec<f64> {
        let start = self.cell(prefix) * self.vocab;
        self.probabilities[start..start + self.vocab].to_vec()
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

#[cfg(test)]
mod tests;
