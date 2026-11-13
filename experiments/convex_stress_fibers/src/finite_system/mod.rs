//! Exact prime-field row operations with explicit work and nullity bounds.
//! The integer normalization idea ports relabel/coding.rs at4510ee7d. This
//! extension handles arbitrary dimensions, affine right sides and retained
//! row-operation certificates; an independent checker replays full rows.
mod audit;
mod solve;
use crate::Result;
pub use audit::check;
use serde::{Deserialize, Serialize};
pub use solve::solve;
#[cfg(test)]
mod tests;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct System {
    pub prime: u16,
    pub variables: usize,
    pub rows: Vec<Vec<u16>>,
    pub right: Vec<u16>,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Policy {
    pub scalar_updates: u64,
    pub row_operations: usize,
    pub nullity: usize,
}
impl Default for Policy {
    fn default() -> Self {
        Self {
            scalar_updates: 500_000_000,
            row_operations: 2_000_000,
            nullity: 3,
        }
    }
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Problem {
    pub system: System,
    pub policy: Policy,
}
pub fn prime(p: u16) -> bool {
    (2..=257).contains(&p)
        && !(2..)
            .take_while(|d| d * d <= p)
            .any(|d| p.is_multiple_of(d))
}
impl Problem {
    pub fn validate(&self) -> Result<u64> {
        let s = &self.system;
        if !prime(s.prime)
            || !(1..=1024).contains(&s.variables)
            || s.rows.is_empty()
            || s.rows.len() > 8192
            || s.right.len() != s.rows.len()
            || s.right.iter().any(|&b| b >= s.prime)
            || s.rows
                .iter()
                .any(|r| r.len() != s.variables || r.iter().any(|&a| a >= s.prime))
            || self.policy.scalar_updates > 1_000_000_000
            || self.policy.row_operations > 2_000_000
            || self.policy.nullity > 8
        {
            return Err("invalid bounded prime-field system or elimination policy".into());
        }
        Ok(512 * 1024 * 1024
            + 8 * s.rows.len() as u64 * (s.variables + 1) as u64
            + 64 * self.policy.row_operations as u64)
    }
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Operation {
    Swap {
        a: usize,
        b: usize,
    },
    Scale {
        row: usize,
        start: usize,
        factor: u16,
    },
    Subtract {
        target: usize,
        source: usize,
        start: usize,
        factor: u16,
    },
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Space {
    pub particular: Vec<u16>,
    pub directions: Vec<Vec<u16>>,
}
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Work {
    pub pivot_tests: usize,
    pub inverse_tests: usize,
    pub scalar_updates: u64,
    pub row_operations: usize,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Outcome {
    pub problem: Problem,
    pub augmented: Vec<Vec<u16>>,
    pub operations: Vec<Operation>,
    pub pivots: Vec<usize>,
    pub free: Vec<usize>,
    pub contradiction: Option<usize>,
    pub space: Option<Space>,
    pub reason: String,
    pub work: Work,
    pub buffer_bound_bytes: u64,
}
pub fn replay(problem: &Problem, outcome: &Outcome) -> Result<()> {
    if &outcome.problem != problem || solve(problem.clone())? != *outcome {
        return Err("finite-system input/state/operation/work replay mismatch".into());
    }
    Ok(())
}
