//! CM27 residual_identity/lift_iff with a bounded bank of actual digits. The
//! modular producer at f661532e retained only the free-zero representative;
//! here every emitted affine point is lifted and checked independently.
use crate::{Result, affine, finite_system};
use convex_stress_modular::System;
use serde::{Deserialize, Serialize};
mod audit;
mod run;
#[cfg(test)]
mod tests;
pub use audit::check;
pub use run::run;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Request {
    pub system: System,
    pub modulus: u16,
    pub prime: u16,
    pub base: Vec<u16>,
    pub elimination: finite_system::Policy,
    pub enumeration: affine::Policy,
}
impl Request {
    fn validate(&self) -> Result<()> {
        self.system.validate_point(self.modulus, &self.base)?;
        if !finite_system::prime(self.prime) || self.modulus as u32 * self.prime as u32 > 257 {
            return Err("branched lift exceeds prime/total modulus257".into());
        }
        Ok(())
    }
    fn bound(&self, field: &affine::Request) -> Result<u64> {
        let rows = self.system.rows.len() as u64;
        let variables = self.system.variables as u64;
        let bound = field.validate()?
            + 512 * 1024 * 1024
            + 128 * (rows * variables + self.enumeration.points as u64 * (rows + variables));
        if bound > 8 * 1024 * 1024 * 1024 {
            return Err("branched lift report/JSON forecast exceeds8GiB".into());
        }
        Ok(bound)
    }
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Branch {
    pub point: Vec<u16>,
    pub original_difference: Vec<i64>,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Work {
    pub base_products: u64,
    pub lifted_products: u64,
    pub ce_gradients: usize,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Report {
    pub input: Request,
    pub residual: Vec<i64>,
    pub digits: affine::Report,
    pub branches: Vec<Branch>,
    pub entire_digit_space: bool,
    pub reason: String,
    pub buffer_preflight_bytes: u64,
    pub work: Work,
}
pub fn replay(input: &Request, out: &Report) -> Result<()> {
    if run(input.clone())? != *out {
        return Err("branched lift original-data/digit/branch/work replay differs".into());
    }
    Ok(())
}
