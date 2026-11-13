//! CM30 / Lean e02b7c49: colliding class codes are legitimate latent proposals.
//! The finite-field engine ports the exact core used by CM27. CM25 bounds the
//! actual complex latent matrices; CD8 trains a convex ordinary-CE head on fixed
//! features. Original row programs certify codes under CM30's assumptions.
//! Discovery is discrete, with two shift gauges and observed categorical names;
//! it supplies no arithmetic answer rule or raw-label identification claim.
pub use convex_stress_consequences::row_program;
pub use convex_stress_fibers::{Result, finite_system};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeSet;
mod audit;
pub mod binding;
pub mod certificate;
pub mod model;
#[cfg(test)]
mod model_tests;
mod plan;
mod run;
#[cfg(test)]
mod tests;
pub mod training;
pub use audit::check;
pub use run::run;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Policy {
    pub equations: usize,
    pub elimination: finite_system::Policy,
}
impl Default for Policy {
    fn default() -> Self {
        Self {
            equations: 4096,
            elimination: finite_system::Policy {
                nullity: 1,
                ..Default::default()
            },
        }
    }
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Input {
    pub n: usize,
    pub vocabulary: usize,
    /// A cell is row+n*column; values are categorical names only.
    pub observations: Vec<[usize; 2]>,
    pub policy: Policy,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Request {
    pub input: Input,
    pub prime: u16,
}
/// The entire supported state catalogue; independent of data or target names.
pub fn catalogue() -> Vec<u16> {
    (2..=257).filter(|&p| finite_system::prime(p)).collect()
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Names {
    pub rows: Vec<usize>,
    pub columns: Vec<usize>,
    pub classes: Vec<usize>,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Code {
    pub prime: u16,
    pub names: Names,
    pub rows: Vec<u16>,
    pub columns: Vec<u16>,
    pub classes: Vec<u16>,
}
impl Code {
    pub fn query(&self, row: usize, column: usize) -> Option<u16> {
        if self.prime < 2 {
            return None;
        }
        let i = self.names.rows.iter().position(|&r| r == row)?;
        let j = self.names.columns.iter().position(|&c| c == column)?;
        Some(
            ((*self.rows.get(i)? as u32 + *self.columns.get(j)? as u32) % self.prime as u32) as u16,
        )
    }
    pub fn label(&self, label: usize) -> Option<u16> {
        self.names
            .classes
            .iter()
            .position(|&c| c == label)
            .and_then(|i| self.classes.get(i).copied())
    }
}
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Work {
    pub field: finite_system::Work,
    pub inverse_tests: usize,
    pub normalization_products: usize,
    pub observation_checks: usize,
    pub ce_gradients: usize,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Report {
    pub request: Request,
    pub buffer_preflight_bytes: u64,
    pub names: Names,
    pub field: finite_system::Outcome,
    pub field_audit: Value,
    pub anchor_class: Option<usize>,
    pub inverse: Option<u16>,
    pub code: Option<Code>,
    pub conflict: Option<usize>,
    pub code_fibers: Vec<Vec<usize>>,
    pub raw_label_identification: bool,
    pub reason: String,
    pub work: Work,
}
pub fn replay(input: &Request, report: &Report) -> Result<()> {
    if run(input.clone())? != *report {
        return Err("original code discovery/state/work does not replay".into());
    }
    Ok(())
}
