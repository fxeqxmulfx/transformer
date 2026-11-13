//! CM28 RowReduction and CM29 decoded_of_program (Lean af4a6d14).
//! Build an actual echelon row program over ZMod q. Divisor pivots are allowed:
//! integer quotients cancel divisible entries without inverting a nonunit.
//! Unit columns are cleared in every row to shorten subsequent query programs.
//! No complete row-module basis or identification theorem is claimed here.
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
    pub normalization_tests: u64,
}
impl Default for Policy {
    fn default() -> Self {
        Self {
            scalar_updates: 2_000_000_000,
            row_operations: 2_000_000,
            pivot_tests: 20_000_000,
            combination_tests: 20_000_000,
            normalization_tests: 1_000_000,
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
                .any(|row| row.len() != self.variables || row.iter().any(|&x| x >= self.modulus))
        {
            return Err("row program exceeds bounded shape or canonical residues".into());
        }
        let p = &self.policy;
        if p.scalar_updates > 4_000_000_000
            || p.row_operations > 2_000_000
            || p.pivot_tests > 40_000_000
            || p.combination_tests > 40_000_000
            || p.normalization_tests > 2_000_000
        {
            return Err("row program policy exceeds declared work limits".into());
        }
        Ok(512 * 1024 * 1024
            + 64 * (self.rows.len() * self.variables) as u64
            + 512 * p.row_operations as u64)
    }
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Op {
    Swap { i: usize, j: usize },
    Scale { i: usize, factor: u16 },
    Add { i: usize, j: usize, factor: u16 },
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Pivot {
    pub row: usize,
    pub column: usize,
    pub divisor: u16,
}
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Work {
    pub scalar_updates: u64,
    pub row_operations: usize,
    pub swapped_coefficients: u64,
    pub pivot_tests: u64,
    pub combination_tests: u64,
    pub normalization_tests: u64,
    pub gcd_steps: u64,
    pub ce_gradients: usize,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Outcome {
    pub input: Problem,
    pub buffer_preflight_bytes: u64,
    pub operations: Vec<Op>,
    pub matrix: Vec<Vec<u16>>,
    pub pivots: Vec<Pivot>,
    pub visited_columns: usize,
    pub reason: String,
    pub work: Work,
}
pub fn replay(input: &Problem, out: &Outcome) -> Result<()> {
    if run(input.clone())? != *out {
        return Err(
            "row program original-input/operation/pivot/control/work replay differs".into(),
        );
    }
    Ok(())
}
