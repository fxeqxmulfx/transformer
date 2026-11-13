//! CM26 / Lean13711438: bounded observed-only proposals, not identification.
//! Each admitted vertex is in CM25's convex spectral hull; discrete discovery
//! is a heuristic within it. Raw numerical spectral audits are not real proofs.
mod audit;
mod candidate;
mod run;
#[cfg(test)]
mod tests;
mod witness;
use crate::{Model, Result, Sample, finite_system, groups};
pub use audit::check;
pub use run::run;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Policy {
    pub guesses: usize,
    pub constraints: usize,
    pub enumeration: usize,
    pub models: usize,
    pub elimination: finite_system::Policy,
}
impl Default for Policy {
    fn default() -> Self {
        Self {
            guesses: 4,
            constraints: 4096,
            enumeration: 65536,
            models: 16,
            elimination: finite_system::Policy::default(),
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
    pub fn samples(&self) -> Vec<Sample> {
        self.observations
            .iter()
            .map(|&[cell, label]| Sample { cell, label })
            .collect()
    }
    pub fn validate(&self) -> Result<u64> {
        convex_stress_tagged::validate_observations(self.n, self.vocab, &self.samples())?;
        let p = &self.policy;
        if p.guesses > 7
            || !(1..=8192 - 3).contains(&p.constraints)
            || p.enumeration > 262144
            || p.models > 32
            || p.elimination.nullity > 3
            || p.elimination.scalar_updates > 1_000_000_000
            || p.elimination.row_operations > 2_000_000
        {
            return Err(
                "fiber search exceeds declared guesses/equations/enumeration/model/work bounds"
                    .into(),
            );
        }
        // Simultaneous report, independent checker and replay, including JSON
        // buffers and full operation traces. No q-by-n-by-n frequency tensor.
        let per = 32 * (p.constraints + 3) as u64 * (2 * self.n + 2) as u64
            + 256 * p.elimination.row_operations as u64
            + 1024 * p.enumeration as u64;
        let bound = 2 * 1024 * 1024 * 1024
            + p.guesses as u64 * per
            + 1024 * p.models as u64 * (self.n * self.n) as u64;
        if bound > 16 * 1024 * 1024 * 1024 {
            return Err("aggregate fiber discovery exceeds sixteen GiB forecast".into());
        }
        Ok(bound)
    }
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Decision {
    Collision {
        axis: groups::Axis,
        first: usize,
        second: usize,
        phase: u16,
    },
    Conflict {
        first: usize,
        second: usize,
        latent: usize,
    },
    Uncovered {
        latent: usize,
    },
    FiberMismatch {
        group: usize,
        cardinality: usize,
        mass: usize,
    },
    Duplicate {
        candidate: usize,
    },
    Accepted {
        candidate: usize,
    },
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Trial {
    pub coefficients: Vec<u16>,
    pub decision: Decision,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Attempt {
    pub size: usize,
    pub group_indices: Vec<usize>,
    pub constraints_truncated: bool,
    pub elimination: Option<finite_system::Outcome>,
    pub elimination_audit: Option<Value>,
    pub total_coefficients: Option<u64>,
    pub trials: Vec<Trial>,
    pub reason: String,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Candidate {
    pub guess: usize,
    pub trial: usize,
    pub common_mass: usize,
    pub model: Model,
    pub audit: Value,
}
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Work {
    pub grouping_cell_visits: usize,
    pub elimination_scalar_updates: u64,
    pub elimination_row_operations: usize,
    pub coefficient_vectors: usize,
    pub affine_products: usize,
    pub permutation_entries: usize,
    pub decoder_observations: usize,
    pub fiber_entries: usize,
    pub raw_frequencies: usize,
    pub ce_gradients: usize,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Report {
    pub input: Input,
    pub buffer_bound_bytes: u64,
    pub partition: groups::Partition,
    pub attempts: Vec<Attempt>,
    pub candidates: Vec<Candidate>,
    pub selected: Option<usize>,
    pub reason: String,
    pub work: Work,
}
pub fn replay(input: &Input, report: &Report) -> Result<()> {
    if &report.input != input || run(input.clone())? != *report {
        return Err("fiber original-input/equation/enumeration/model/work replay mismatch".into());
    }
    Ok(())
}
