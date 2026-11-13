//! CM28 / Lean6a8fb49e: uniqueness is conditional on a fixed categorical view,
//! including its unknown bijective interior and any constant-strip hypothesis.
use convex_stress_identified::{categorical, ring};
use convex_stress_modular::{Result, search};
use convex_stress_relabel::hash;
use serde::{Deserialize, Serialize};
use serde_json::Value;
mod binding;
mod run;
mod selection;
#[cfg(test)]
mod tests;
pub use run::{check, replay, run};
pub use selection::select;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Policy {
    pub retained_equations: usize,
    pub ring: ring::Policy,
}
impl Default for Policy {
    fn default() -> Self {
        Self {
            retained_equations: 4096,
            ring: ring::Policy::default(),
        }
    }
}
impl Policy {
    pub fn validate(&self) -> Result<()> {
        if !(1..=4096).contains(&self.retained_equations) {
            return Err("identification equation policy exceeds4096".into());
        }
        ring::Problem {
            modulus: 2,
            variables: 1,
            rows: vec![vec![0]],
            policy: self.ring.clone(),
        }
        .validate()?;
        Ok(())
    }
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Scope {
    pub kind: String,
    pub unknown_bijective_interior_required: bool,
    pub constant_strip_structure_required: bool,
    pub constant: Option<usize>,
    pub row_map: Vec<Option<usize>>,
    pub column_map: Vec<Option<usize>>,
    pub unconditional_ground_truth_identification: bool,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Entry {
    pub candidate: usize,
    pub view: usize,
    pub model_sha256: String,
    pub request_sha256: String,
    pub full_prediction_checks: usize,
    pub scope: Scope,
    pub certificate: categorical::Report,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Report {
    pub policy: Policy,
    pub parent_audit: Value,
    pub parent_views: Vec<Value>,
    pub entries: Vec<Entry>,
    pub certificate_count: usize,
    pub buffer_preflight_bytes: u64,
    pub ce_gradients: usize,
}
