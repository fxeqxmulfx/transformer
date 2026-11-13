//! CM27 / Lean98b12abc: learn unknown categorical codes in residue components.
//! One-dimensional component kernels and unique normalized digits are required
//! by this proposal policy. Refused branches do not refute the larger hull.
mod audit;
mod base;
mod component;
mod component_audit;
mod endpoint;
mod plan;
mod run;
#[cfg(test)]
mod tests;
use crate::{Model, Result, Sample, System, finite_system, lift, recombine};
pub use audit::check;
pub use run::run;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Policy {
    pub observation_equations: usize,
    pub scalar_updates_per_stage: u64,
    pub total_scalar_updates: u64,
    pub row_operations_per_stage: usize,
    pub stages: usize,
}
impl Default for Policy {
    fn default() -> Self {
        Self {
            observation_equations: 4096,
            scalar_updates_per_stage: 500_000_000,
            total_scalar_updates: 2_000_000_000,
            row_operations_per_stage: 2_000_000,
            stages: 8,
        }
    }
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Input {
    pub n: usize,
    pub vocab: usize,
    pub observations: Vec<[usize; 2]>,
    pub policy: Policy,
}
impl Input {
    pub fn validate(&self) -> Result<u64> {
        if !(2..=257).contains(&self.n)
            || !(2..=1024).contains(&self.vocab)
            || self.observations.is_empty()
            || self.observations.len() > self.n * self.n
        {
            return Err("invalid bounded categorical code input".into());
        }
        let mut seen = vec![false; self.n * self.n];
        for &[cell, label] in &self.observations {
            if cell >= seen.len() || label >= self.vocab || seen[cell] {
                return Err("invalid/repeated categorical cell or class".into());
            }
            seen[cell] = true;
        }
        let p = &self.policy;
        if !(1..=4096).contains(&p.observation_equations)
            || p.scalar_updates_per_stage > 1_000_000_000
            || p.total_scalar_updates > 4_000_000_000
            || p.row_operations_per_stage > 2_000_000
            || p.stages > 16
        {
            return Err("modular code equation/stage/work policy exceeds limits".into());
        }
        let variables = 2 * self.n + 257;
        let rows = p.observation_equations + 2 * self.n + 3;
        let bound = 2 * 1024 * 1024 * 1024
            + p.stages as u64
                * (32 * (rows * variables) as u64 + 256 * p.row_operations_per_stage as u64);
        if bound > 16 * 1024 * 1024 * 1024 {
            return Err("modular code report/check/replay forecast exceeds16GiB".into());
        }
        Ok(bound)
    }
    fn samples(&self) -> Vec<Sample> {
        self.observations
            .iter()
            .map(|&[cell, label]| Sample { cell, label })
            .collect()
    }
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Partition {
    pub classes: Vec<usize>,
    pub class_indices: Vec<usize>,
    pub missing_rows: Vec<usize>,
    pub missing_columns: Vec<usize>,
    pub anchor_row: usize,
    pub retained_observations: usize,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Factor {
    pub prime: u16,
    pub exponent: usize,
    pub modulus: u16,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Base {
    pub prime: u16,
    pub field: finite_system::Outcome,
    pub audit: Value,
    pub anchor_class: Option<usize>,
    pub inverse: Option<u16>,
    pub inverse_tests: usize,
    pub point: Option<Vec<u16>>,
    pub reason: String,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Component {
    pub factor: Factor,
    pub base: Option<Base>,
    pub anchored_system: Option<System>,
    pub lifts: Vec<lift::Step>,
    pub point: Option<Vec<u16>>,
    pub reason: String,
}
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Work {
    pub finite_stages: usize,
    pub finite_scalar_updates: u64,
    pub finite_row_operations: usize,
    pub finite_pivot_tests: usize,
    pub finite_inverse_tests: usize,
    pub inverse_tests: usize,
    pub normalization_products: usize,
    pub base_original_products: usize,
    pub lift_original_products: usize,
    pub crt_original_products: usize,
    pub observation_checks: usize,
    pub raw_frequencies: usize,
    pub ce_gradients: usize,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Report {
    pub input: Input,
    pub buffer_bound_bytes: u64,
    pub partition: Partition,
    pub system: Option<System>,
    pub factors: Vec<Factor>,
    pub components: Vec<Component>,
    pub crt: Vec<recombine::Step>,
    pub crt_audits: Vec<Value>,
    pub point: Option<Vec<u16>>,
    pub collision: Option<[usize; 3]>,
    pub observation_conflict: Option<usize>,
    pub model: Option<Model>,
    pub audit: Option<Value>,
    pub reason: String,
    pub work: Work,
}
pub fn replay(input: &Input, out: &Report) -> Result<()> {
    if &out.input != input || run(input.clone())? != *out {
        return Err(
            "categorical code original-input/component/lift/CRT/work replay mismatch".into(),
        );
    }
    Ok(())
}
