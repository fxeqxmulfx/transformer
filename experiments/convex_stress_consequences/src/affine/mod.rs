//! Enumerate a bounded prefix of an actual checked affine space. Unlike the
//! task222 rank-one proposal, multiple free coordinates are retained. Every
//! emitted point must satisfy all original equations; no uniqueness is claimed.
use crate::{Result, finite_system};
use serde::{Deserialize, Serialize};
use serde_json::Value;
mod audit;
mod run;
#[cfg(test)]
mod tests;
pub use audit::check;
pub use run::run;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Policy {
    pub points: usize,
    pub coordinate_products: u64,
    pub original_products: u64,
}
impl Default for Policy {
    fn default() -> Self {
        Self {
            points: 256,
            coordinate_products: 500_000_000,
            original_products: 500_000_000,
        }
    }
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Request {
    pub problem: finite_system::Problem,
    pub policy: Policy,
}
impl Request {
    pub fn validate(&self) -> Result<u64> {
        let p = &self.policy;
        if p.points > 65536
            || p.coordinate_products > 2_000_000_000
            || p.original_products > 2_000_000_000
        {
            return Err("affine enumeration policy exceeds declared caps".into());
        }
        let bound = 512 * 1024 * 1024
            + 4 * self.problem.validate()?
            + 64 * p.points as u64 * (self.problem.system.variables + 8) as u64;
        if bound > 8 * 1024 * 1024 * 1024 {
            return Err("affine report/check/replay forecast exceeds8GiB".into());
        }
        Ok(bound)
    }
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Point {
    pub ordinal: usize,
    pub weights: Vec<u16>,
    pub values: Vec<u16>,
}
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Work {
    pub points: usize,
    pub parameter_digits: usize,
    pub coordinate_products: u64,
    pub original_products: u64,
    pub ce_gradients: usize,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Report {
    pub input: Request,
    pub buffer_preflight_bytes: u64,
    pub elimination: finite_system::Outcome,
    pub elimination_audit: Value,
    /// None means no completed space, or the exact cardinality exceeds u64.
    pub cardinality: Option<u64>,
    pub cardinality_overflow: bool,
    pub points: Vec<Point>,
    pub complete: bool,
    pub reason: String,
    pub work: Work,
}
pub fn replay(input: &Request, out: &Report) -> Result<()> {
    if run(input.clone())? != *out {
        return Err("affine original-input/space/point/order/work replay differs".into());
    }
    Ok(())
}
