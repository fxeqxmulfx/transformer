//! CM12: fixed output reindexing around the existing coupled spectral solver.
//! Lean 0c89afcc proves CE covariance and the invariant direct-prediction hull.
//! Discrete coding proposals are conditional candidates, not a global oracle.
pub use convex_stress_budgets::{
    C, Model as Table, Result, Sample, budgets, check_layout, choose_by_train, data, gradient,
    hash, ideal_loss, linalg, loss, metrics, projection, random, samples,
    solver::{Mode, Solver},
};
pub mod coding;
pub mod cohort;
pub use coding::{Coding, Discovery};
use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize)]
pub struct Model {
    pub coding: Coding,
    pub table: Table,
}
impl Model {
    pub fn validate_shape(&self) -> Result<()> {
        self.coding.validate(self.table.vocab)?;
        self.table.validate_shape()
    }
}
impl metrics::Predictor for Model {
    fn vocab(&self) -> usize {
        self.table.vocab
    }
    fn logits(&self, prefix: &[i64]) -> Vec<f64> {
        let start = self.table.cell(prefix) * self.table.vocab;
        self.coding
            .encode
            .iter()
            .map(|&encoded| self.table.probabilities[start + encoded])
            .collect()
    }
}

/// A conservative explicit-buffer bound before allocating an inherited dense
/// solver: 32 real/complex table-sized work buffers plus 64 MiB overhead.
/// The launcher additionally limits the entire native process to 8 GiB.
pub fn memory_bound(n: usize) -> Result<u64> {
    if !(2..=193).contains(&n) {
        return Err("model vocabulary exceeds frozen 193-class cap".into());
    }
    Ok(32 * 8 * (n as u64).pow(3) + 64 * 1024 * 1024)
}

#[cfg(test)]
mod tests;
