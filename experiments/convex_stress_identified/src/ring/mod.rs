//! Lean CompositeDiscovery.RowReduction: the checker applies complete rows
//! using i128, independently of the producer's bounded u32 arithmetic. Replay
//! additionally binds deterministic pivot decisions and control-work counters.
use crate::Result;
use serde::{Deserialize, Serialize};
mod audit;
mod run;
#[cfg(test)]
mod tests;
pub use audit::check;
pub use run::run;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Policy {
    pub scalar_updates: u64,
    pub row_operations: usize,
    pub pivot_tests: u64,
    pub combination_tests: u64,
    pub inverse_tests: u64,
}
impl Default for Policy {
    fn default() -> Self {
        Self {
            scalar_updates: 2_000_000_000,
            row_operations: 2_000_000,
            pivot_tests: 20_000_000,
            combination_tests: 20_000_000,
            inverse_tests: 1_000_000,
        }
    }
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Problem {
    pub modulus: u16,
    pub variables: usize,
    pub rows: Vec<Vec<u16>>,
    pub policy: Policy,
}
impl Problem {
    pub fn validate(&self) -> Result<u64> {
        if !(2..=257).contains(&self.modulus)
            || !(1..=1024).contains(&self.variables)
            || self.rows.is_empty()
            || self.rows.len() > 8192
            || self
                .rows
                .iter()
                .any(|r| r.len() != self.variables || r.iter().any(|&x| x >= self.modulus))
        {
            return Err("ring matrix exceeds bounded dimensions or canonical residues".into());
        }
        let p = &self.policy;
        if p.scalar_updates > 4_000_000_000
            || p.row_operations > 2_000_000
            || p.pivot_tests > 40_000_000
            || p.combination_tests > 40_000_000
            || p.inverse_tests > 2_000_000
        {
            return Err("ring work policy exceeds declared limits".into());
        }
        // Includes producer/checker/replay matrices, instruction and JSON copies.
        let forecast = 512 * 1024 * 1024
            + 64 * (self.rows.len() * self.variables) as u64
            + 512 * p.row_operations as u64;
        if forecast > 8 * 1024 * 1024 * 1024 {
            return Err("ring certificate forecast exceeds8GiB".into());
        }
        Ok(forecast)
    }
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Op {
    Swap { i: usize, j: usize },
    Scale { i: usize, factor: u16 },
    Add { i: usize, j: usize, factor: u16 },
}
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Work {
    pub scalar_updates: u64,
    pub row_operations: usize,
    pub swapped_coefficients: u64,
    pub pivot_tests: u64,
    pub combination_tests: u64,
    pub inverse_tests: u64,
    pub gcd_steps: u64,
    pub ce_gradients: usize,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Outcome {
    pub input: Problem,
    pub buffer_bound_bytes: u64,
    pub operations: Vec<Op>,
    pub matrix: Vec<Vec<u16>>,
    pub pivots: Vec<usize>,
    pub complete: bool,
    pub reason: String,
    pub work: Work,
}
pub fn replay(input: &Problem, out: &Outcome) -> Result<()> {
    if run(input.clone())? != *out {
        return Err("ring input, pivot, operation, state, stop or work replay differs".into());
    }
    Ok(())
}
