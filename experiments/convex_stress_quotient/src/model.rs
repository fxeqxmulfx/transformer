//! CM25 actual latent Fourier bounds; CD8 fixed-feature convex readout.
//! The discovery step is discrete and is not part of the convex CE problem.
use crate::{Report, Result};
use convex_stress_tagged::{C, linalg, metrics};
use serde::{Deserialize, Serialize};
mod audit;
mod initialize;
pub use audit::Audit;
pub use initialize::{Initialization, check_initialization, initialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Model {
    pub n: usize,
    pub q: usize,
    pub vocabulary: usize,
    pub rows: Vec<u16>,
    pub columns: Vec<u16>,
    pub known_rows: Vec<bool>,
    pub known_columns: Vec<bool>,
    pub latent: Vec<u16>,
    /// One combined tensor W[z,c], not separately learned routing/decoder factors.
    pub coefficients: Vec<f64>,
    pub radius: f64,
}
pub fn memory_bound(n: usize, q: usize, v: usize) -> Result<u64> {
    if !(2..=257).contains(&n) || !(2..=257).contains(&q) || !(2..=1024).contains(&v) {
        return Err("latent model dimensions exceed n257/q257/vocabulary1024".into());
    }
    // Raw SVD workspaces stream one n*n complex matrix. Includes serialization,
    // count/gradient copies and the independent head replay, not n*n*q*v tensors.
    Ok(512 * 1024 * 1024 + 2048 * (n * n) as u64 + 256 * (q * v) as u64)
}
impl Model {
    pub fn validate(&self) -> Result<()> {
        memory_bound(self.n, self.q, self.vocabulary)?;
        if self.rows.len() != self.n
            || self.columns.len() != self.n
            || self.known_rows.len() != self.n
            || self.known_columns.len() != self.n
            || self.latent.len() != self.n * self.n
            || self.coefficients.len() != self.q * self.vocabulary
            || self.radius != 32. * self.vocabulary as f64
            || self
                .rows
                .iter()
                .chain(&self.columns)
                .any(|&x| x as usize >= self.q)
            || self.coefficients.iter().any(|x| !x.is_finite())
        {
            return Err("invalid saved latent/readout shape or finite budget".into());
        }
        for (cell, &z) in self.latent.iter().enumerate() {
            if z as u32
                != (self.rows[cell % self.n] as u32 + self.columns[cell / self.n] as u32)
                    % self.q as u32
            {
                return Err("saved latent table differs from actual phase coordinates".into());
            }
        }
        if (0..self.vocabulary).any(|c| {
            self.coefficients
                .iter()
                .skip(c)
                .step_by(self.vocabulary)
                .map(|x| x.abs())
                .sum::<f64>()
                > self.radius + 1e-8
        }) {
            return Err("combined head exceeds its per-class L1 bound".into());
        }
        Ok(())
    }
    pub fn logits(&self, cell: usize) -> &[f64] {
        let offset = self.latent[cell] as usize * self.vocabulary;
        &self.coefficients[offset..offset + self.vocabulary]
    }
    pub fn predictions(&self) -> Vec<usize> {
        (0..self.latent.len())
            .map(|cell| {
                let logits = self.logits(cell);
                (1..self.vocabulary)
                    .fold(0, |best, c| if logits[c] > logits[best] { c } else { best })
            })
            .collect()
    }
    pub fn known(&self, cell: usize) -> bool {
        self.known_rows[cell % self.n] && self.known_columns[cell / self.n]
    }
    pub(crate) fn observations(&self, samples: &[[usize; 2]]) -> Result<()> {
        self.validate()?;
        if samples.is_empty() || samples.len() > self.n * self.n {
            return Err("invalid original observation count".into());
        }
        let mut seen = vec![false; self.n * self.n];
        for &[cell, label] in samples {
            if cell >= seen.len() || label >= self.vocabulary || seen[cell] {
                return Err("invalid or repeated original cell/raw label".into());
            }
            seen[cell] = true;
        }
        Ok(())
    }
}
impl metrics::Predictor for Model {
    fn vocab(&self) -> usize {
        self.vocabulary
    }
    fn logits(&self, prefix: &[i64]) -> Vec<f64> {
        assert!(prefix.len() >= 2);
        let (r, c) = (prefix[prefix.len() - 1], prefix[prefix.len() - 2]);
        assert!(r >= 0 && c >= 0 && (r as usize) < self.n && (c as usize) < self.n);
        self.logits(r as usize + self.n * c as usize).to_vec()
    }
}
